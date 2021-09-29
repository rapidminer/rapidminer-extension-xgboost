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

import java.util.function.BooleanSupplier;

import ml.dmlc.xgboost4j.java.DMatrix;
import ml.dmlc.xgboost4j.java.XGBoostError;


/**
 * Wrapper for {@link DMatrix} instances that allows to block invocations of {@link #getHandle()} using a sentinel. The
 * handle is only returned if the given sentinel returns {@code true}.
 * <p>
 * This wrapper only exists to add support for exiting the boosting loop (aborting the training) which is not supported
 * by the XGBoost4J API.
 *
 * @author Michael Knopf
 */
class CheckedDMatrix extends DMatrix {

	private BooleanSupplier sentinel;

	public CheckedDMatrix(float[] data, int rows, int columns) throws XGBoostError {
		super(data, rows, columns);
		this.sentinel = null;
	}

	/**
	 * Invocations of {@link #getHandle()} will be blocked if the given sentinel is not {@code null} and returns {@code
	 * false}.
	 */
	public void setSentinel(BooleanSupplier sentinel) {
		this.sentinel = sentinel;
	}

	/**
	 * @return the native matrix handle
	 * @throws UsageBlockedException if the current sentinel returns {@code false}
	 */
	@Override
	public long getHandle() {
		if (sentinel == null || sentinel.getAsBoolean()) {
			return super.getHandle();
		} else {
			throw new UsageBlockedException();
		}
	}

	static class UsageBlockedException extends RuntimeException {
		UsageBlockedException() {
			super("XGBoost was blocked from acquiring DMatrix handle");
		}
	}

}
