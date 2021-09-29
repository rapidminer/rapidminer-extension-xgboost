# RapidMiner XGBoost Extension

This extension embeds the [XGBoost eXtreme Gradient Boosting](https://github.com/dmlc/xgboost) library for use in RapidMiner.
It implements a single operator named _XGBoost_ compatible with RapidMiner's builtin learners.

## Features

* **Automatic data conversion**
  
  Automatic data conversion between RapidMiner tables and XGBoost's internal format.
  Supports all data types except datetime values.
* **Encoding of categorical data**

  Automatic encoding of categorical features using binary columns or a modified one-hot encoding.
  The user can of course always implement their own encoding using RapidMiner operators.
* **Objective selection**
  
  Automatic selection of the learning objective based on the input data.
  Supports regression, binary classification, and multi-categorical classification problems.
  The objective can also be specified explicitly by the user.

## Limitations

* **No GPU support**

  The distribution of XGBoost used by this extension lacks GPU support.
  Thus, the extension only provides the CPU backends of XGBoost.
* **Limited concurrency**
  
  The extension limits the concurrent use of XGBoost learners due to observed instabilities of some XGBoost APIs.
  Take note that a single XGBoost learner will still scale out to multiple CPU cores.
  However, running multiple XGBoost learners in parallel on a many-core system might underutilize the CPU.

## License

The extension code is provided  under the terms of the [GNU Affero General Public License](./LICENSE).

## Where to start?

The extension can be installed from the [RapidMiner marketplace](https://marketplace.rapidminer.com/UpdateServer/faces/product_details.xhtml?productId=rmx_xgboost).
It includes tutorial processes that explain the basic usage.
For the most part the extension exposes the same parameters as the Python an R packages.
Thus, you can refer to the [official documentation](https://xgboost.readthedocs.io/en/latest/parameter.html) for details.

If you are interested in the code,
a good starting point is the implementation of the [XGBoostWrapper](./src/main/java/com/rapidminer/extension/xgboost/model/XGBoostWrapper.java),
which includes most of the conversion logic.
The corresponding [unit tests](./src/test/java/com/rapidminer/extension/xgboost/model/XGBoostWrapperTests.java) might also be of help.