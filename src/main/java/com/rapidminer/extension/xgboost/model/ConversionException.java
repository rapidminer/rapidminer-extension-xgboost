package com.rapidminer.extension.xgboost.model;

import java.io.IOException;


/**
 * Signals that an exception has occurred when converting an input table to XGBoost's representation.
 *
 * @author Michael Knopf
 */
public class ConversionException extends IOException {

	ConversionException(String message) {
		super(message);
	}

}
