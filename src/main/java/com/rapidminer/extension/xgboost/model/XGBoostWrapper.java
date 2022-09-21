/*
 * RapidMiner GmbH
 *
 * Copyright (C) 2021-2021 by RapidMiner GmbH and the contributors
 *
 * Complete list of developers available at our web site:
 *
 *      www.rapidminer.com
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License
 * along with this program.  If not, see http://www.gnu.org/licenses/.
 */
package com.rapidminer.extension.xgboost.model;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.BooleanSupplier;
import java.util.stream.IntStream;

import com.rapidminer.adaption.belt.IOTable;
import com.rapidminer.belt.buffer.Buffers;
import com.rapidminer.belt.buffer.NominalBuffer;
import com.rapidminer.belt.buffer.NumericBuffer;
import com.rapidminer.belt.column.Column;
import com.rapidminer.belt.column.Columns;
import com.rapidminer.belt.column.Dictionary;
import com.rapidminer.belt.reader.CategoricalReader;
import com.rapidminer.belt.reader.NumericReader;
import com.rapidminer.belt.reader.Readers;
import com.rapidminer.belt.table.Table;
import com.rapidminer.belt.util.ColumnRole;
import com.rapidminer.example.AttributeWeights;

import ml.dmlc.xgboost4j.java.Booster;
import ml.dmlc.xgboost4j.java.DMatrix;
import ml.dmlc.xgboost4j.java.XGBoost;
import ml.dmlc.xgboost4j.java.XGBoostError;


/**
 * Wrapper for the XGBoost bindings that takes care of converting RapidMiner {@link Table}s to XGBoost {@link DMatrix}
 * instances.
 * <p>
 * XGBoost supports missing values but does not support categorical features out of the box. The wrapper converts
 * categorical columns into one of the following two formats: if the column has at most two classes, the column is
 * converted into a single float vector with 0 and 1 representing the negative and positive class respectively. If the
 * column has no boolean mapping, the class with the higher index is assumed to be the positive class. Missing value are
 * encoded as {@link Float#NaN}.
 * <p>
 * If the column has more than two classes, a modified one-hot encoding is applied: class vectors are encoded using
 * values {@link Float#NaN} and 1 instead of the more common 0 and 1. In other words, they are encoded as unary instead
 * of binary features.
 *
 * @author Michael Knopf
 */
public class XGBoostWrapper {

	/** JVM-wide lock for native XGBoost methods that are not thread-safe. */
	private static final Object XGB_LOCK = new Object();

	private XGBoostWrapper() {
		throw new AssertionError("Static utility class must not be initialized");
	}

	/**
	 * Trains a new {@link XGBoostModel} on the given data.
	 */
	public static XGBoostModel train(Table data, Table validation, Map<String, String> parameters, int iterations,
									 int earlyStopping, BooleanSupplier sentinel)
			throws XGBoostError, ConversionException {
		if (data.height() == 0) {
			throw new IllegalArgumentException("Training table must not be empty");
		}

		Map<String, float[]> trainingMatrices = createTrainingMatrices(data);
		Map<String, float[]> validationMatrices = validation == null ? null : createTrainingMatrices(validation);

		Map<String, String> trainingParameters = new HashMap<>(parameters);
		selectObjective(data, trainingParameters);

		synchronized (XGB_LOCK) {
			CheckedDMatrix matrix = toMatrix(trainingMatrices);
			matrix.setSentinel(sentinel);
			Map<String, DMatrix> watches = validationMatrices == null
					? Collections.emptyMap()
					: Collections.singletonMap("validation", toMatrix(validationMatrices));

			try {
				Booster booster = XGBoost.train(matrix, new HashMap<>(trainingParameters), iterations, watches,
						null, null, null, earlyStopping);
				int completedIterations = booster.getVersion() / 2;
				byte[] model = booster.toByteArray();

				// Booster is no longer used after this point.
				booster.dispose();

				return new XGBoostModel(new IOTable(data), trainingParameters, completedIterations, model);
			} catch(CheckedDMatrix.UsageBlockedException e) {
				// Boosting aborted by sentinel.
				return null;
			} finally {
				// Matrix is no longer used after this point.
				matrix.dispose();
			}
		}
	}

	/**
	 * Extracts the "total_gain" feature importance scores from the given model.
	 *
	 * @param model the wrapped booster
	 * @param table the reference features
	 * @return the "total_gain" feature importance scores
	 * @throws XGBoostError if the score lookup fails
	 * @throws IOException  if the XGBoost deserialization fails
	 */
	public static AttributeWeights getWeights(XGBoostModel model, IOTable table) throws XGBoostError, IOException {
		Map<String, Double> scores;
		synchronized (XGB_LOCK) {
			Booster booster = XGBoost.loadModel(model.getBooster());
			scores = booster.getScore(model.getTrainingHeader().getTable().labels().toArray(new String[0]), "total_gain");
			booster.dispose();
		}

		AttributeWeights weights = new AttributeWeights(table);
		for (String label : weights.getAttributeNames()) {
			weights.setWeight(label, scores.getOrDefault(label, 0.0));
		}

		return weights;
	}

	/**
	 * Applies the given model to the given features and returns the prediction as new {@link Column}. If available,
	 * class scores are added to the given map.
	 *
	 * @throws IOException if the XGBoost deserialization fails
	 */
	public static Column predict(XGBoostModel model, Table features, Map<String, Column> scores)
			throws XGBoostError, IOException {
		if (features.height() == 0) {
			throw new IllegalArgumentException("Scoring table must not be empty");
		}

		float[] featureMatrix = XGBoostWrapper.createFeatureMatrix(features);
		int width = featureMatrix.length / features.height();
		int height = features.height();

		float[][] predictions;
		synchronized (XGB_LOCK) {
				Booster booster = XGBoost.loadModel(model.getBooster());
				DMatrix matrix = new DMatrix(featureMatrix, height, width);
				predictions = booster.predict(matrix);
				// Do not wait for GC to free native resources.
				matrix.dispose();
				booster.dispose();
		}

		Column label = model.getLabelColumn();
		if (label.type().category() == Column.Category.CATEGORICAL) {
			return Columns.isAtMostBicategorical(label)
					? predictBicategorical(predictions, label, scores)
					: predictCategorical(predictions, label, scores);
		} else {
			return predictRegression(predictions);
		}
	}

	private static void selectObjective(Table data, Map<String, String> parameters) {
		if (!parameters.containsKey("objective")) {
			Column label = data.select().withMetaData(ColumnRole.LABEL).columns().get(0);
			if (label.type().category() == Column.Category.CATEGORICAL) {
				if (Columns.isAtMostBicategorical(label)) {
					parameters.put("objective", "binary:logistic");
				} else {
					parameters.put("objective", "multi:softprob");
					parameters.put("num_class", Integer.toString(label.getDictionary().size()));
				}
			} else {
				parameters.put("objective", "reg:squarederror");
			}
		}
	}

	private static CheckedDMatrix toMatrix(Map<String, float[]> matrices) throws XGBoostError {
		float[] features = matrices.get("features");
		float[] label = matrices.get("label");

		int height = label.length;
		int width = features.length / height;

		CheckedDMatrix matrix = new CheckedDMatrix(features, height, width);
		matrix.setLabel(label);
		if (matrices.containsKey("weights")) {
			matrix.setWeight(matrices.get("weights"));
		}

		return matrix;
	}

	private static Column predictBicategorical(float[][] predictions, Column label, Map<String, Column> scores) {
		Dictionary dictionary = label.getDictionary();
		int negativeIndex = dictionary.isBoolean() ?
				dictionary.getNegativeIndex() :
				IntStream.range(1, dictionary.maximalIndex())
						.filter(i -> dictionary.get(i) != null)
						.findFirst().orElse(-1);
		int positiveIndex = dictionary.isBoolean() ? dictionary.getPositiveIndex() : dictionary.maximalIndex();

		String negativeValue = dictionary.get(negativeIndex);
		String positiveValue = dictionary.get(positiveIndex);

		NominalBuffer clazz = Buffers.nominalBuffer(predictions.length);
		NumericBuffer negativeScore = Buffers.realBuffer(predictions.length, false);
		NumericBuffer positiveScore = Buffers.realBuffer(predictions.length, false);
		for (int i = 0; i < predictions.length; i++) {
			float score = predictions[i][0];
			clazz.set(i, score < 0.5 ? negativeValue : positiveValue);
			negativeScore.set(i, 1.0 - score);
			positiveScore.set(i, score);
		}

		// There is no negative value if there is only one class.
		if (negativeValue != null) {
			scores.put(negativeValue, negativeScore.toColumn());
		}
		scores.put(positiveValue, positiveScore.toColumn());

		return Columns.changeDictionary(clazz.toColumn(), label);
	}

	private static Column predictCategorical(float[][] predictions, Column label, Map<String, Column> scores) {
		int nClasses = predictions[0].length;
		Dictionary dictionary = label.getDictionary();
		NominalBuffer clazz = Buffers.nominalBuffer(predictions.length);
		NumericBuffer[] scoreBuffers = new NumericBuffer[nClasses];
		Arrays.setAll(scoreBuffers, i -> Buffers.realBuffer(predictions.length, false));

		for (int y = 0; y < predictions.length; y++) {
			float[] row = predictions[y];
			float maxScore = Float.NEGATIVE_INFINITY;
			int maxIndex = -1;
			for (int x = 0; x < nClasses; x++) {
				float score = row[x];
				if (score > maxScore) {
					maxScore = score;
					maxIndex = x;
				}
				scoreBuffers[x].set(y, score);
			}
			clazz.set(y, dictionary.get(maxIndex + 1));
		}

		for (int i = 0; i < nClasses; i++) {
			String value = dictionary.get(i + 1);
			if (value != null) {
				scores.put(value, scoreBuffers[i].toColumn());
			}
		}

		return Columns.changeDictionary(clazz.toColumn(), label);
	}

	private static Column predictRegression(float[][] predictions) {
		NumericBuffer predictionBuffer = Buffers.realBuffer(predictions.length, false);
		for (int y = 0; y < predictions.length; y++) {
			predictionBuffer.set(y, predictions[y][0]);
		}
		return predictionBuffer.toColumn();
	}

	private static float[] createFeatureMatrix(Table table) throws ConversionException {
		List<Column> potentialFeatures = table.select()
				.withoutMetaData(ColumnRole.class)
				.columns();

		if (potentialFeatures.isEmpty()) {
			throw new IllegalArgumentException("Data table does not contain any feature");
		}

		List<Column> numericFeatures = new ArrayList<>();
		List<Column> categoricalFeatures = new ArrayList<>();
		long encodedFeatures = 0;

		for (Column column : potentialFeatures) {
			switch (column.type().category()) {
				case NUMERIC:
					numericFeatures.add(column);
					encodedFeatures++;
					break;
				case CATEGORICAL:
					categoricalFeatures.add(column);
					Dictionary dictionary = column.getDictionary();
					encodedFeatures += Columns.isAtMostBicategorical(column) ? 1 : dictionary.size();
					break;
				default:
					// Ignore unsupported columns.
					break;
			}
		}

		// Conservative estimate of the maximum supported array size (see OpenJDK's ArraysSupport).
		if (encodedFeatures * table.height() > Integer.MAX_VALUE - 8) {
			throw new ConversionException("Size of encoded data set exceeds runtime limit");
		}

		int width = (int) encodedFeatures;
		float[] features = new float[width * table.height()];
		Arrays.fill(features, Float.NaN);
		int x = 0;

		for (Column column : numericFeatures) {
			readNumericColumn(column, features, x, width);
			x++;
		}

		for (Column column : categoricalFeatures) {
			if (Columns.isAtMostBicategorical(column)) {
				readBicategoricalColumn(column, features, x, width);
				x++;
			} else {
				Dictionary dictionary = column.getDictionary();
				readCategoricalColumn(column, features, x, width);
				x += dictionary.size();
			}
		}

		return features;
	}

	private static Map<String, float[]> createTrainingMatrices(Table table) throws ConversionException {
		Map<String, float[]> matrix = new HashMap<>();
		matrix.put("features", createFeatureMatrix(table));

		List<Column> weightColumns = table.select().withMetaData(ColumnRole.WEIGHT).columns();
		if (!weightColumns.isEmpty()) {
			float[] weights = new float[table.height()];
			readNumericColumn(weightColumns.get(0), weights, 0, 1);
			matrix.put("weights", weights);
		}

		List<Column> labelColumns = table.select().withMetaData(ColumnRole.LABEL).columns();
		if (labelColumns.isEmpty()) {
			throw new IllegalArgumentException("Input table has no label");
		}

		Column labelColumn = labelColumns.get(0);
		float[] label = new float[table.height()];

		switch (labelColumn.type().category()) {
			case NUMERIC:
				readNumericColumn(labelColumn, label, 0, 1);
				break;
			case CATEGORICAL:
				if (Columns.isAtMostBicategorical(labelColumn)) {
					readBicategoricalColumn(labelColumn, label, 0, 1);
				} else {
					readNumericColumn(labelColumn, label, 0, 1);
					for (int i = 0; i < label.length; i++) {
						label[i] -= 1;
					}
				}
				break;
			default:
				throw new IllegalArgumentException("Unsupported label column");
		}

		matrix.put("label", label);
		return matrix;
	}

	private static void readNumericColumn(Column column, float[] destination, int offset, int step) {
		NumericReader reader = Readers.numericReader(column);
		int i = offset;
		while (reader.hasRemaining()) {
			float value = (float) reader.read();
			destination[i] = value;
			i += step;
		}
	}

	private static void readBicategoricalColumn(Column column, float[] destination, int offset, int step) {
		Dictionary dictionary = column.getDictionary();
		int negative = dictionary.isBoolean() ?
				dictionary.getNegativeIndex() :
				IntStream.range(1, dictionary.maximalIndex())
						.filter(i -> dictionary.get(i) != null)
						.findFirst().orElse(-1);
		int positive = dictionary.isBoolean() ? dictionary.getPositiveIndex() : dictionary.maximalIndex();

		CategoricalReader reader = Readers.categoricalReader(column);
		int i = offset;
		while (reader.hasRemaining()) {
			int index = reader.read();
			if (index == negative) {
				destination[i] = 0;
			} else if (index == positive) {
				destination[i] = 1;
			}
			i += step;
		}
	}

	private static void readCategoricalColumn(Column column, float[] destination, int offset, int step) {
		Dictionary dictionary = column.getDictionary();
		if (dictionary.size() == 0) {
			// Nothing to do
			return;
		}

		int[] indexMap = new int[dictionary.maximalIndex() + 1];
		int counter = 0;
		for (int i = 1; i <= dictionary.maximalIndex(); i++) {
			if (dictionary.get(i) != null) {
				indexMap[i] = counter;
				counter++;
			}
		}

		CategoricalReader reader = Readers.categoricalReader(column);
		int i = offset;
		while (reader.hasRemaining()) {
			int index = reader.read();
			if (index > 0) {
				destination[i + indexMap[index]] = 1;
			}
			i += step;
		}
	}

}
