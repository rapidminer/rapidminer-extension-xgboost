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

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

import java.io.IOException;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.function.BooleanSupplier;

import org.junit.Test;

import com.rapidminer.adaption.belt.IOTable;
import com.rapidminer.belt.column.Column;
import com.rapidminer.belt.column.Dictionary;
import com.rapidminer.belt.execution.Context;
import com.rapidminer.belt.execution.SequentialContext;
import com.rapidminer.belt.execution.Workload;
import com.rapidminer.belt.reader.CategoricalReader;
import com.rapidminer.belt.reader.Readers;
import com.rapidminer.belt.table.Builders;
import com.rapidminer.belt.table.Table;
import com.rapidminer.belt.util.ColumnRole;
import com.rapidminer.example.AttributeWeights;
import com.rapidminer.example.set.TableSplitter;
import com.rapidminer.operator.UserError;

import ml.dmlc.xgboost4j.java.XGBoostError;


public class XGBoostWrapperTests {

	private static final Context CTX = new SequentialContext();

	@Test
	public void testRegression() throws XGBoostError, IOException {
		Table data = Builders.newTableBuilder(100)
				.addReal("A", i -> i)
				.addReal("B", i -> 2 * i)
				.addReal("C", i -> 3 * i)
				.addReal("Label", i -> 4 * i)
				.addMetaData("Label", ColumnRole.LABEL)
				.build(CTX);

		XGBoostModel model = XGBoostWrapper.train(data, null, Collections.emptyMap(), 10, 0, () -> true);

		assertNotNull(model);
		assertEquals(10, model.getIterations());
		assertEquals("reg:squarederror", model.getParameters().get("objective"));

		Map<String, Column> scores = new HashMap<>();
		Column column = XGBoostWrapper.predict(model, data, scores);
		assertEquals(Column.TypeId.REAL, column.type().id());
		assertTrue(scores.isEmpty());
	}

	@Test
	public void testOneClassLabel() throws XGBoostError, IOException {
		// The label is trivial to predict to ensure 100% accuracy.
		Random rng = new Random(123456);
		Table data = Builders.newTableBuilder(100)
				.addReal("A", i -> (i % 2) * 5 + rng.nextDouble())
				.addReal("B", i -> rng.nextDouble())
				.addReal("C", i -> rng.nextDouble())
				.addNominal("Label", i -> i % 2 == 0 ? "A" : null)
				.addMetaData("Label", ColumnRole.LABEL)
				.build(CTX);

		XGBoostModel model = XGBoostWrapper.train(data, null, Collections.singletonMap("seed", "123456"),
				100, 0, () -> true);
		assertNotNull(model);

		Map<String, Column> scores = new HashMap<>();
		Column column = XGBoostWrapper.predict(model, data, scores);
		assertEquals(Column.TypeId.NOMINAL, column.type().id());
		assertEquals(Collections.singleton("A"), scores.keySet());

		String[] label = new String[100];
		String[] prediction = new String[100];
		data.column("Label").fill(label, 0);
		column.fill(prediction, 0);
		assertArrayEquals(label, prediction);
	}

	@Test
	public void testBooleanClassification() throws XGBoostError, IOException {
		Random rng = new Random(123456);
		Table data = Builders.newTableBuilder(100)
				.addReal("A", i -> i)
				.addReal("B", i -> (i % 2) * 5 + rng.nextDouble())
				.addReal("C", i -> 3 * i)
				.addBoolean("Label", i -> i % 2 == 0 ? "False" : "True", "True")
				.addMetaData("Label", ColumnRole.LABEL)
				.build(CTX);

		XGBoostModel model = XGBoostWrapper.train(data, null, Collections.emptyMap(), 10, 0, () -> true);

		assertNotNull(model);
		assertEquals(10, model.getIterations());
		assertEquals("binary:logistic", model.getParameters().get("objective"));

		Map<String, Column> scores = new HashMap<>();
		Column column = XGBoostWrapper.predict(model, data, scores);
		assertEquals(Column.TypeId.NOMINAL, column.type().id());

		assertTrue(column.getDictionary().isBoolean());
		assertEquals(new HashSet<>(Arrays.asList("True", "False")), scores.keySet());
	}

	@Test
	public void testPredictionOfSingleClass() throws XGBoostError, IOException {
		Table data = Builders.newTableBuilder(100)
				.addReal("A", i -> i)
				.addReal("B", i -> 3 * i)
				.addBoolean("Label", i -> i % 2 == 0 ? "False" : "True", "True")
				.addMetaData("Label", ColumnRole.LABEL)
				.build(CTX);

		XGBoostModel model = XGBoostWrapper.train(data, null, Collections.emptyMap(), 10, 0, () -> true);

		assertNotNull(model);
		assertEquals(10, model.getIterations());
		assertEquals("binary:logistic", model.getParameters().get("objective"));

		// The test data contains a single row for which the prediction is positive.
		Table testData = data.rows(1, 2, CTX);
		Map<String, Column> scores = new HashMap<>();
		Column column = XGBoostWrapper.predict(model, testData, scores);
		assertEquals(Column.TypeId.NOMINAL, column.type().id());

		assertTrue(column.getDictionary().isBoolean());
		Dictionary dictionary = column.getDictionary();
		assertEquals("True", dictionary.get(dictionary.getPositiveIndex()));
		assertEquals("False", dictionary.get(dictionary.getNegativeIndex()));
		assertEquals(new HashSet<>(Arrays.asList("True", "False")), scores.keySet());
	}

	@Test
	public void testBinaryClassification() throws XGBoostError, IOException {
		Random rng = new Random(123456);
		Table data = Builders.newTableBuilder(100)
				.addReal("A", i -> i)
				.addReal("B", i -> (i % 2) * 5 + rng.nextDouble())
				.addReal("C", i -> 3 * i)
				.addNominal("Label", i -> i % 2 == 0 ? "False" : "True")
				.addMetaData("Label", ColumnRole.LABEL)
				.build(CTX);

		XGBoostModel model = XGBoostWrapper.train(data, null, Collections.emptyMap(), 10, 0, () -> true);

		assertNotNull(model);
		assertEquals(10, model.getIterations());
		assertEquals("binary:logistic", model.getParameters().get("objective"));

		Map<String, Column> scores = new HashMap<>();
		Column column = XGBoostWrapper.predict(model, data, scores);
		assertEquals(Column.TypeId.NOMINAL, column.type().id());

		assertFalse(column.getDictionary().isBoolean());
		assertEquals(new HashSet<>(Arrays.asList("True", "False")), scores.keySet());
	}

	@Test
	public void testClassification() throws XGBoostError, IOException {
		String[] dictionary = {"One", "Two", "Three", "Four", "Five"};
		Random rng = new Random(123456);
		Table data = Builders.newTableBuilder(100)
				.addReal("A", i -> i)
				.addReal("B", i -> (i % 5) * 10 + rng.nextDouble())
				.addReal("C", i -> 3 * i)
				.addNominal("Label", i -> dictionary[i % 5])
				.addMetaData("Label", ColumnRole.LABEL)
				.build(CTX);

		XGBoostModel model = XGBoostWrapper.train(data, null, Collections.emptyMap(), 10, 0, () -> true);

		assertNotNull(model);
		assertEquals(10, model.getIterations());
		assertEquals("multi:softprob", model.getParameters().get("objective"));

		Map<String, Column> scores = new HashMap<>();
		Column column = XGBoostWrapper.predict(model, data, scores);
		assertEquals(Column.TypeId.NOMINAL, column.type().id());

		assertFalse(column.getDictionary().isBoolean());
		assertEquals(new HashSet<>(Arrays.asList(dictionary)), scores.keySet());
	}

	@Test
	public void testClassificationWithSparseDictionary() throws XGBoostError, IOException {
		// The label is trivial to predict to ensure all classes are predicted.
		String[] dictionary = {"One", "Two", "Three", "Four", "Five"};
		Random rng = new Random(123456);
		Table data = Builders.newTableBuilder(100)
				.addReal("A", i -> i)
				.addReal("B", i -> (i % 5) * 10 + rng.nextDouble())
				.addReal("C", i -> 3 * i)
				.addNominal("Label", i -> dictionary[i % 5])
				.addMetaData("Label", ColumnRole.LABEL)
				.build(CTX);

		// Exclude one category from the training data:
		int indexOfTwo = data.column("Label").getDictionary().createInverse().get("Two");
		data = data.filterCategorical("Label", i -> i != indexOfTwo, Workload.SMALL, CTX);

		XGBoostModel model = XGBoostWrapper.train(data, null, Collections.singletonMap("seed", "123465"),
				10, 0, () -> true);
		assertNotNull(model);

		Map<String, Column> scores = new HashMap<>();
		Column column = XGBoostWrapper.predict(model, data, scores);
		assertEquals(Column.TypeId.NOMINAL, column.type().id());

		CategoricalReader reader = Readers.categoricalReader(column);
		Dictionary predictionDictionary = column.getDictionary();
		Set<String> predictedValues = new HashSet<>();
		while (reader.hasRemaining()) {
			predictedValues.add(predictionDictionary.get(reader.read()));
		}

		// We should get four predicted values but scores for all five classes.
		assertEquals(new HashSet<>(Arrays.asList("One", "Three", "Four", "Five")), predictedValues);
		assertEquals(new HashSet<>(Arrays.asList(dictionary)), scores.keySet());
	}

	@Test
	public void testCustomObjective() throws XGBoostError, ConversionException {
		Table data = Builders.newTableBuilder(100)
				.addReal("A", i -> i)
				.addReal("B", i -> 2 * i)
				.addReal("C", i -> 3 * i)
				.addReal("Label", i -> 4 * i)
				.addMetaData("Label", ColumnRole.LABEL)
				.build(CTX);

		XGBoostModel model = XGBoostWrapper.train(data, null, Collections.singletonMap("objective", "reg:tweedie"),
				10, 0, () -> true);

		assertNotNull(model);
		assertEquals("reg:tweedie", model.getParameters().get("objective"));
	}

	@Test
	public void testWeights() throws XGBoostError, IOException {
		// The weights should favor predictions of class "Y".
		String[] dictionary = {"X", "Y", "Z"};
		double[] weights = {0.05, 0.9, 0.1};
		Random rng = new Random(123456);
		Table data = Builders.newTableBuilder(100)
				.addReal("A", i -> rng.nextDouble())
				.addReal("Weight", i -> weights[i % 3])
				.addMetaData("Weight", ColumnRole.WEIGHT)
				.addNominal("Label", i -> dictionary[i % 3])
				.addMetaData("Label", ColumnRole.LABEL)
				.build(CTX);

		XGBoostModel model = XGBoostWrapper.train(data, null, Collections.singletonMap("seed", "12345"),
				10, 0, () -> true);
		assertNotNull(model);

		Map<String, Column> scores = new HashMap<>();
		Column column = XGBoostWrapper.predict(model, data, scores);

		CategoricalReader reader = Readers.categoricalReader(column);
		Dictionary predictionDictionary = column.getDictionary();
		Set<String> predictedValues = new HashSet<>();
		while (reader.hasRemaining()) {
			predictedValues.add(predictionDictionary.get(reader.read()));
		}

		assertEquals(Collections.singleton("Y"), predictedValues);
		assertEquals(new HashSet<>(Arrays.asList("X", "Y", "Z")), scores.keySet());
	}

	@Test
	public void testMissingValues() throws XGBoostError, IOException {
		String[] dictionary = {"One", "Two", "Three", "Four", "Five"};
		Table table = Builders.newTableBuilder(100)
				.addReal("A", i -> i % 17 == 0 ? Double.NaN : 2 * i)
				.addNominal("B", i -> i % 11 == 0 ? null : dictionary[i % 5])
				.addBoolean("C", i -> i % 7 == 0 ? null : i % 2 == 0 ? "False" : "True", "True")
				.addReal("D", i -> Double.NaN)
				.addNominal("E", i -> null)
				.addReal("Label", i -> 10 * i)
				.addMetaData("Label", ColumnRole.LABEL)
				.build(CTX);

		XGBoostModel model = XGBoostWrapper.train(table, null, Collections.emptyMap(), 10, 0, () -> true);
		assertNotNull(model);

		Column column = XGBoostWrapper.predict(model, table, new HashMap<>());
		assertNotNull(column);
	}

	@Test
	public void testAbortTraining() throws XGBoostError, ConversionException {
		Table table = Builders.newTableBuilder(100)
				.addReal("A", i -> i)
				.addReal("B", i -> 2 * i)
				.addReal("C", i -> 3 * i)
				.addReal("Label", i -> 4 * i)
				.addMetaData("Label", ColumnRole.LABEL)
				.build(CTX);

		int[] counter = {0};
		BooleanSupplier abort = () -> {
				counter[0]++;
				return false;
		};

		// The training should abort immediately and return no model.
		XGBoostModel model = XGBoostWrapper.train(table, null, Collections.emptyMap(), 100, 0, abort);
		assertEquals(1, counter[0]);
		assertNull(model);
	}

	@Test
	public void testEarlyStopping() throws XGBoostError, UserError, ConversionException {
		// The label is trivial to predict to ensure early stopping is possible.
		Random rng = new Random(123456);
		Table table = Builders.newTableBuilder(300)
				.addReal("A", i -> (i % 2) * 5 + rng.nextDouble())
				.addReal("B", i -> rng.nextDouble())
				.addReal("C", i -> rng.nextDouble())
				.addBoolean("Label", i -> i % 2 == 0 ? "A" : "B", "A")
				.addMetaData("Label", ColumnRole.LABEL)
				.build(CTX);

		TableSplitter splitter = new TableSplitter(table, 0.7, TableSplitter.AUTOMATIC, false, 0);
		Table data = splitter.selectSingleSubset(0, CTX);
		Table validationData = splitter.selectSingleSubset(1, CTX);

		XGBoostModel model = XGBoostWrapper.train(data, validationData, Collections.singletonMap("seed", "123465"),
				100, 5, () -> true);
		assertNotNull(model);
		assertTrue(model.getIterations() < 100);
	}

	@Test(expected = ConversionException.class)
	public void testMatrixSizeLimit() throws ConversionException, XGBoostError {
		// One-hot encoding of features A and B results in more than 2^31 values.
		Table table = Builders.newTableBuilder(35000)
				.addNominal("A", String::valueOf)
				.addNominal("B", String::valueOf)
				.addReal("Label", i -> i)
				.addMetaData("Label", ColumnRole.LABEL)
				.build(CTX);

		XGBoostWrapper.train(table, null, Collections.emptyMap(), 10, 0, () -> true);
	}

	@Test(expected = IllegalArgumentException.class)
	public void testMissingLabel() throws XGBoostError, ConversionException {
		Table table = Builders.newTableBuilder(100)
				.addReal("A", i -> i)
				.addReal("B", i -> i)
				.addReal("C", i -> i)
				.build(CTX);

		XGBoostWrapper.train(table, null, Collections.emptyMap(), 100, 0, () -> true);
	}

	@Test(expected = IllegalArgumentException.class)
	public void testMissingFeatures() throws XGBoostError, ConversionException {
		Table table = Builders.newTableBuilder(100)
				.addReal("Label", i -> 4 * i)
				.addMetaData("Label", ColumnRole.LABEL)
				.build(CTX);

		XGBoostWrapper.train(table, null, Collections.emptyMap(), 100, 0, () -> true);
	}

	@Test(expected = IllegalArgumentException.class)
	public void testMetaDataOnly() throws XGBoostError, ConversionException {
		Table table = Builders.newTableBuilder(100)
				.addReal("A", i -> i)
				.addMetaData("A", ColumnRole.METADATA)
				.addReal("B", i -> i)
				.addMetaData("B", ColumnRole.METADATA)
				.addReal("C", i -> i)
				.addMetaData("C", ColumnRole.METADATA)
				.addReal("Label", i -> 4 * i)
				.addMetaData("Label", ColumnRole.LABEL)
				.build(CTX);

		XGBoostWrapper.train(table, null, Collections.emptyMap(), 100, 0, () -> true);
	}

	@Test
	public void testClassificationImportanceWeights() throws XGBoostError, IOException {
		// Column B should always have the highest importance.
		String[] dictionary = {"One", "Two", "Three", "Four", "Five"};
		Random rng = new Random(123456);
		Table data = Builders.newTableBuilder(100)
				.addReal("A", i -> rng.nextDouble())
				.addReal("B", i -> (i % 5) * 10 + rng.nextDouble())
				.addReal("C", i -> 3 * i)
				.addNominal("Label", i -> dictionary[i % 5])
				.addMetaData("Label", ColumnRole.LABEL)
				.build(CTX);

		XGBoostModel model = XGBoostWrapper.train(data, null, Collections.emptyMap(), 100, 0, () -> true);
		assertNotNull(model);

		AttributeWeights weights = XGBoostWrapper.getWeights(model, new IOTable(data));
		assertEquals(AttributeWeights.NO_SORTING, weights.getSortingType());
		assertEquals(AttributeWeights.ORIGINAL_WEIGHTS, weights.getSortingType());

		assertEquals(3, weights.size());
		assertTrue(weights.getWeight("A") < weights.getWeight("B"));
		assertTrue(weights.getWeight("C") < weights.getWeight("B"));
	}

	@Test
	public void testRegressionImportanceWeights() throws XGBoostError, IOException {
		// Column C should always have the highest importance.
		Random rng = new Random(123456);
		Table data = Builders.newTableBuilder(100)
				.addReal("A", i -> rng.nextDouble())
				.addReal("B", i -> (i % 5) * 10)
				.addReal("C", i -> 3 * i + rng.nextDouble())
				.addReal("Label", i -> i)
				.addMetaData("Label", ColumnRole.LABEL)
				.build(CTX);

		XGBoostModel model = XGBoostWrapper.train(data, null, Collections.emptyMap(), 100, 0, () -> true);
		assertNotNull(model);

		AttributeWeights weights = XGBoostWrapper.getWeights(model, new IOTable(data));
		assertEquals(AttributeWeights.NO_SORTING, weights.getSortingType());
		assertEquals(AttributeWeights.ORIGINAL_WEIGHTS, weights.getSortingType());

		assertEquals(3, weights.size());
		assertTrue(weights.getWeight("A") < weights.getWeight("C"));
		assertTrue(weights.getWeight("B") < weights.getWeight("C"));
	}

}
