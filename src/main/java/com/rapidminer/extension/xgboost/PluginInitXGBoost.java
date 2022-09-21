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
package com.rapidminer.extension.xgboost;

import com.rapidminer.extension.xgboost.model.XGBoostModel;
import com.rapidminer.gui.MainFrame;
import com.rapidminer.repository.versioned.JsonStorableIOObjectResolver;


@SuppressWarnings("unused")
public final class PluginInitXGBoost {

	private PluginInitXGBoost() {}

	public static void initPlugin() {
		JsonStorableIOObjectResolver.INSTANCE.register(XGBoostModel.class);
	}

	public static void initGui(MainFrame mainframe) {}

	public static void initFinalChecks() {}

	public static void initPluginManager() {}

	public static Boolean useExtensionTreeRoot() {
		return false;
	}

}
