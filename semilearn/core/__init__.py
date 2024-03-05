
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from .algorithmbase import AlgorithmBase, ImbAlgorithmBase, SupConLossWeights #PERSO : permet l'import de SupConLossWeights dans les dossiers cousins
from .utils.registry import import_all_modules_for_register

import_all_modules_for_register()