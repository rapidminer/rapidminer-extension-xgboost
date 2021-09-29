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
import java.util.Collections;
import java.util.Map;

import com.rapidminer.adaption.belt.IOTable;

import com.rapidminer.belt.column.Column;
import com.rapidminer.belt.table.Table;
import com.rapidminer.belt.table.Tables;
import com.rapidminer.operator.Operator;
import com.rapidminer.operator.OperatorException;
import com.rapidminer.operator.UserError;
import com.rapidminer.operator.learner.IOTablePredictionModel;

import ml.dmlc.xgboost4j.java.XGBoostError;


/**
 * Wraps a native XGBoost model. This wrapper does not reference any native XGBoost code nor does is store any reference
 * to native objects. All interactions with native code are delegated to the {@link XGBoostWrapper}.
 *
 * @author Michael Knopf
 */
public class XGBoostModel extends IOTablePredictionModel {

	private final Map<String, String> parameters;
	private final int iterations;
	/** Serialized XGBoost booster. */
	private final byte[] booster;

	/**
	 * Default constructor for deserialization.
	 */
	@SuppressWarnings("unused")
	public XGBoostModel() {
		this.parameters = Collections.emptyMap();
		this.iterations = 0;
		this.booster = null;
	}

	/**
	 * Wraps the serialized XGBoost booster along with its parameters.
	 */
	public XGBoostModel(IOTable training, Map<String, String> parameters, int iterations, byte[] booster) {
		super(training, Tables.ColumnSetRequirement.EQUAL, Tables.TypeRequirement.REQUIRE_MATCHING_TYPES);
		this.parameters = parameters;
		this.iterations = iterations;
		this.booster = booster;
	}

	@Override
	protected Column performPrediction(Table features, Map<String, Column> confidences, Operator operator)
			throws OperatorException {
		if (features.height() == 0) {
			return getLabelColumn();
		}
		try {
			return XGBoostWrapper.predict(this, features, confidences);
		}  catch (ConversionException e) {
			throw new UserError(null, e, "xgboost.conversion_error", e.getMessage());
		} catch (XGBoostError | IOException e) {
			throw new UserError(null, e, "xgboost.generic_error", e.getMessage());
		}
	}

	@Override
	public String getName() {
		return "XGBoost";
	}

	@Override
	public String toString() {
		StringBuilder builder = new StringBuilder("XGBoost prediction model for label '")
				.append(getLabelName())
				.append("'.\n\nTraining hyper parameters: \n\n");
		parameters.forEach((key, value) -> builder.append(key).append(" = ").append(value).append("\n"));
		builder.append("\nBoosting iterations: ")
				.append(iterations);
		return builder.toString();
	}

	@Override
	public Column getLabelColumn() {
		return super.getLabelColumn();
	}

	public Map<String, Object> getParameters() {
		return Collections.unmodifiableMap(parameters);
	}

	public int getIterations() {
		return iterations;
	}

	public byte[] getBooster() {
		return booster;
	}

}
