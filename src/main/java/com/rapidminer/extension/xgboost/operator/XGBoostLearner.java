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
package com.rapidminer.extension.xgboost.operator;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.EnumSet;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import com.rapidminer.adaption.belt.IOTable;
import com.rapidminer.belt.execution.Context;
import com.rapidminer.belt.table.Table;
import com.rapidminer.belt.table.Tables;
import com.rapidminer.example.set.TableSplitter;
import com.rapidminer.extension.xgboost.model.ConversionException;
import com.rapidminer.extension.xgboost.model.XGBoostWrapper;
import com.rapidminer.operator.IOTableModel;
import com.rapidminer.operator.Operator;
import com.rapidminer.operator.OperatorDescription;
import com.rapidminer.operator.OperatorException;
import com.rapidminer.operator.TableCapability;
import com.rapidminer.operator.UserError;
import com.rapidminer.operator.learner.AbstractIOTableLearner;
import com.rapidminer.operator.ports.InputPort;
import com.rapidminer.operator.ports.metadata.table.TablePrecondition;
import com.rapidminer.parameter.ParameterType;
import com.rapidminer.parameter.ParameterTypeCategory;
import com.rapidminer.parameter.ParameterTypeDouble;
import com.rapidminer.parameter.ParameterTypeEnumeration;
import com.rapidminer.parameter.ParameterTypeInt;
import com.rapidminer.parameter.ParameterTypeString;
import com.rapidminer.parameter.ParameterTypeTupel;
import com.rapidminer.parameter.UndefinedParameterError;
import com.rapidminer.parameter.conditions.EqualStringCondition;
import com.rapidminer.parameter.conditions.NonEqualStringCondition;
import com.rapidminer.parameter.conditions.ParameterCondition;
import com.rapidminer.tools.LogService;
import com.rapidminer.tools.RandomGenerator;
import com.rapidminer.tools.belt.BeltErrorTools;
import com.rapidminer.tools.belt.BeltTools;

import ml.dmlc.xgboost4j.java.XGBoostError;

import static com.rapidminer.operator.TableCapability.*;


/**
 * Operator that wraps XGBoost as RapidMiner learner. This class is mostly concerned with implementing RapidMiner's
 * frontend interfaces, e.g., defining parameters exposed to the user. See {@link XGBoostWrapper} for the backend
 * implementation that embeds the actual XGBoost library.
 *
 * @author Michael Knopf
 */
public class XGBoostLearner extends AbstractIOTableLearner {

	private static final String PARAMETER_ROUNDS = "rounds";
	private static final String PARAMETER_EARLY_STOPPING = "early_stopping";
	private static final String PARAMETER_EARLY_STOPPING_ROUNDS = "early_stopping_rounds";
	private static final String PARAMETER_EXPERT = "expert_parameters";

	/** Parameters that do not correspond to a named XGBoost hyper-parameter. */
	private static final Set<String> META_PARAMETERS = new HashSet<>(Arrays.asList(
			PARAMETER_ROUNDS,
			PARAMETER_EARLY_STOPPING,
			PARAMETER_EARLY_STOPPING_ROUNDS,
			PARAMETER_EXPERT
	));

	private static final EnumSet<TableCapability> CAPABILITIES = EnumSet.of(
			NOMINAL_COLUMNS,
			TWO_CLASS_COLUMNS,
			NUMERIC_COLUMNS,
			TIME_COLUMNS,
			MISSING_VALUES,
			NOMINAL_LABEL,
			NUMERIC_LABEL,
			ONE_CLASS_LABEL,
			TWO_CLASS_LABEL,
			MISSINGS_IN_LABEL,
			WEIGHTED_ROWS
	);

	private static final EnumSet<TableCapability> UNSUPPORTED = EnumSet.of(
			DATE_TIME_COLUMNS,
			ADVANCED_COLUMNS,
			NO_LABEL,
			MULTIPLE_LABELS,
			UPDATABLE
	);

	private static final Map<String, String> PARAMETER_ALIASES = new HashMap<>();
	static {
		PARAMETER_ALIASES.put("tree booster", "gbtree");
		PARAMETER_ALIASES.put("linear booster", "gblinear");
		PARAMETER_ALIASES.put("DART", "dart");
		PARAMETER_ALIASES.put("approximate", "approx");
		PARAMETER_ALIASES.put("histogram", "hist");
	}

	private InputPort validationSet;

	public XGBoostLearner(OperatorDescription description) {
		super(description);
		this.validationSet = null;
		getParameters().addObserver((observable, parameter) -> {
			if (PARAMETER_EARLY_STOPPING.equals(parameter)) {
				checkValidationPort();
			}
		}, false);
	}

	@Override
	public Operator cloneOperator(String name, boolean forParallelExecution) {
		XGBoostLearner clone = (XGBoostLearner) super.cloneOperator(name, forParallelExecution);
		if (validationSet != null) {
			// Cloning the operator will not trigger the parameter observer.
			clone.enableValidationPort();
		}
		return clone;
	}

	private void checkValidationPort() {
		boolean useValidationSet = false;
		try {
			useValidationSet = "custom".equals(getParameterAsString(PARAMETER_EARLY_STOPPING));
		} catch (UndefinedParameterError e) {
			LogService.getRoot().warning("XGBoost: Failed to look up parameter early stopping.");
		}
		if (useValidationSet) {
			enableValidationPort();
		} else if (validationSet != null) {
			getInputPorts().removePort(validationSet);
			validationSet = null;
		}
	}

	private void enableValidationPort() {
		if (validationSet == null) {
			this.validationSet = getInputPorts().createPort("validation");
			TablePrecondition optionalTable = new TablePrecondition(this.validationSet);
			this.validationSet.addPrecondition(optionalTable);
		}
	}

	@Override
	public IOTableModel learn(IOTable trainingTable) throws OperatorException {
		Context context = BeltTools.getContext(this);

		Table data = trainingTable.getTable();
		Table validationData;
		int earlyStoppingRounds;

		switch (getParameterAsString(PARAMETER_EARLY_STOPPING)) {
			case "auto":
				TableSplitter splitter = new TableSplitter(data, 0.7, TableSplitter.AUTOMATIC, false, 0);
				data = splitter.selectSingleSubset(0, context);
				validationData = splitter.selectSingleSubset(1, context);
				earlyStoppingRounds = getParameterAsInt(PARAMETER_EARLY_STOPPING_ROUNDS);
				break;
			case "custom":
				IOTable validationContainer = validationSet == null ? null : validationSet.getDataOrNull(IOTable.class);
				if (validationContainer != null) {
					validationData = Tables.adapt(validationContainer.getTable(), trainingTable.getTable(),
							Tables.ColumnHandling.REORDER, Tables.DictionaryHandling.CHANGE);
				} else {
					throw new UserError(this, "xgboost.missing_validation_set");
				}
				try {
					// User errors thrown by this method reference model application (misleading).
					BeltErrorTools.requireCompatibleRegulars(null, validationData, trainingTable.getTable(),
							Tables.ColumnSetRequirement.EQUAL, Tables.TypeRequirement.REQUIRE_MATCHING_TYPES);
				} catch (UserError e) {
					throw new UserError(this, e, "xgboost.incompatible_validation_set");
				}
				earlyStoppingRounds = getParameterAsInt(PARAMETER_EARLY_STOPPING_ROUNDS);
				break;
			case "none":
			default:
				validationData = null;
				earlyStoppingRounds = 0;
				break;
		}

		try {
			IOTableModel model =  XGBoostWrapper.train(data, validationData, compileModelParameters(),
					getParameterAsInt(PARAMETER_ROUNDS), earlyStoppingRounds, context::isActive);
			// Check whether the training was aborted.
			checkForStop();
			return model;
		} catch (ConversionException e) {
			throw new UserError(null, e, "xgboost.conversion_error", e.getMessage());
		} catch (XGBoostError e) {
			throw new UserError(null, e, "xgboost.generic_error", e.getMessage());
		}
	}

	private Map<String, String> compileModelParameters() throws UndefinedParameterError {
		Map<String, String> parameters = new HashMap<>();

		// Only include visible parameters.
		for (ParameterType type: getParameters().getParameterTypes()) {
			if (!type.isHidden()) {
				String key = type.getKey();
				if (!META_PARAMETERS.contains(key) && isParameterSet(key)) {
					String value = getParameterAsString(type.getKey());
					parameters.put(type.getKey(), PARAMETER_ALIASES.getOrDefault(value, value));
				}
			}
		}

		// The XGBoost methods are synchronized, thus pass on the parallelism level to the algorithm.
		parameters.put("nthread", Integer.toString(BeltTools.getContext(this).getParallelism()));

		// Silent operation:
		parameters.put("verbosity", "0");

		// Derive random seed from process random generator:
		parameters.put("seed", Integer.toUnsignedString(RandomGenerator.getRandomGenerator(this).nextInt()));

		// Add expert parameters last to allow overriding defaults chosen above.
		for (String parameter: ParameterTypeEnumeration.transformString2Enumeration(
				getParameterAsString(PARAMETER_EXPERT))) {
			String[] pair = ParameterTypeTupel.transformString2Tupel(parameter);
			String key = pair[0];
			String value = pair[1];
			if (value != null && !value.isEmpty()) {
				parameters.put(key, value);
			}
		}

		return parameters;
	}

	@Override
	public Set<TableCapability> supported() {
		return CAPABILITIES;
	}

	@Override
	public Set<TableCapability> unsupported() {
		return UNSUPPORTED;
	}

	@Override
	public List<ParameterType> getParameterTypes() {
		List<ParameterType> types = new ArrayList<>();

		// Add types in the order they are listed in the documentation.
		types.add(new ParameterTypeCategory("booster", "The boosting algorithm to use.",
				new String[]{"tree booster", "linear booster", "DART"}, 0, false));

		types.add(new ParameterTypeInt(PARAMETER_ROUNDS, "The maximum number of boosting rounds.", 1,
				Integer.MAX_VALUE, 25));

		types.add(new ParameterTypeCategory(PARAMETER_EARLY_STOPPING,
				"Controls the optional early stopping of boosting iterations.",
				new String[]{"none", "auto", "custom"}, 0, false));
		ParameterType type = new ParameterTypeInt(PARAMETER_EARLY_STOPPING_ROUNDS,
				"Stop the model training if the model performance does not improve " +
				"after the given number of boosting rounds.", 0, Integer.MAX_VALUE, 10);
		type.registerDependencyCondition(new NonEqualStringCondition(this, PARAMETER_EARLY_STOPPING, false, "none"));
		types.add(type);

		// The following group of parameters is used by both tree boosters:
		ParameterCondition treeBooster = new EqualStringCondition(this, "booster", false, "tree booster", "DART");

		List<ParameterType> group = new ArrayList<>();
		group.add(new ParameterTypeDouble("learning_rate",
				"Step size shrinkage used after each boosting round to prevent overfitting.", 0, 1, 0.3));
		group.add(new ParameterTypeDouble("min_split_loss",
				"Minimum loss reduction required to further partition a leaf node of the tree.", 0,
				Double.POSITIVE_INFINITY, 0));
		group.add(new ParameterTypeInt("max_depth", "Maximum depth of a tree.", 0, Integer.MAX_VALUE, 6));
		group.add(new ParameterTypeDouble("min_child_weight",
				"Minimum sum of instance weights (hessian) to further partition a leaf node of the tree.", 0,
				Double.POSITIVE_INFINITY, 1));
		group.add(new ParameterTypeDouble("subsample",
				"Trains trees on sub-samples of the training data of the given size.", 0, 1, 1));
		group.add(new ParameterTypeCategory("tree_method", "The tree construction algorithm used in XGBoost.",
				new String[]{"auto", "exact", "approximate", "histogram"}, 0, false));

		group.forEach(t -> t.registerDependencyCondition(treeBooster));
		types.addAll(group);
		group.clear();

		// Used by all boosters:
		types.add(new ParameterTypeDouble("lambda", "L2 regularization term on weights.",
				Double.NEGATIVE_INFINITY, Double.POSITIVE_INFINITY, 1));
		types.add(new ParameterTypeDouble("alpha", "L1 regularization term on weights.",
				Double.NEGATIVE_INFINITY, Double.POSITIVE_INFINITY, 0));

		// Parameters used only for the DART booster:
		group.add(new ParameterTypeCategory("sample_type", "The sampling algorithm.",
				new String[]{"uniform", "weighted"}, 0, false));
		group.add(new ParameterTypeCategory("normalize_type", "The normalization algorithm.",
				new String[]{"tree", "forest"}, 0, false));
		group.add(new ParameterTypeDouble("rate_drop",
				"Dropout rate (the fraction of previous trees to drop).", 0, 1, 0));
		group.add(new ParameterTypeDouble("skip_drop",
				"Probability of skipping the dropout procedure during a boosting round.", 0, 1, 0));

		group.forEach(t -> t.registerDependencyCondition(new EqualStringCondition(this, "booster", false, "DART")));
		types.addAll(group);
		group.clear();

		// Parameters for the linear booster:
		group.add(new ParameterTypeCategory("updater", "Fitting algorithm for the linear model.",
				new String[]{"shotgun", "coord_descent"}, 0, false));
		group.add(new ParameterTypeCategory("feature_selector", "Feature selection and ordering method.",
				new String[]{"cyclic", "shuffle", "random", "greedy", "thrifty"}, 0, false));

		type = new ParameterTypeInt("top_k",
				"The number of top features to select in the greedy and thrifty feature selectors.", 0, Integer.MAX_VALUE, 0);
		type.registerDependencyCondition(new EqualStringCondition(this, "feature_selector", false, "greedy", "thrifty"));
		group.add(type);

		group.forEach(t -> t.registerDependencyCondition(new EqualStringCondition(this, "booster", false, "linear booster")));
		types.addAll(group);
		group.clear();

		types.forEach(t -> t.setExpert(false));

		// Manual expert parameters
		ParameterType expertParameterKey = new ParameterTypeCategory("key", "", new String[]{
				// Learning task parameters
				"objective",
				"base_score",
				"eval_metric",
				// Tree Booster parameters
				"max_delta_step",
				"sampling_method",
				"colsample_bytree",
				"colsample_bylevel",
				"colsample_bynode",
				"sketch_eps",
				"scale_pos_weight",
				"updater",
				"refresh_leaf",
				"grow_policy",
				"max_leaves",
				"max_bin",
				"predictor",
				"num_parallel_tree",
				"monotone_constraints",
				"interaction_constraints",
				"single_precision_histogram",
				"deterministic_histogram",
				// DART parameters
				"one_drop",
				// Tweedie regression parameters
				"tweedie_variance_power"
			}, 0);

		ParameterType expertParameter = new ParameterTypeTupel("parameter", "", expertParameterKey,
				new ParameterTypeString("value", "", "", true));
		types.add(new ParameterTypeEnumeration(PARAMETER_EXPERT, "", expertParameter, true));

		return types;
	}
}
