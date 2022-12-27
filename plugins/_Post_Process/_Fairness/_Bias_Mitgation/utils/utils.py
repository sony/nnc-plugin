# Copyright 2022 Sony Group Corporation.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from sklearn.metrics import confusion_matrix


def get_demographic_parity(y_privileged, y_unprivileged,
                           preds_privileged, preds_unprivileged):
    """
    Computed as the difference between the rate of positive outcomes
    received by the unprivileged group to the privileged group.
    Args:
        y_privileged (numpy.ndarray) : actual privileged class
        y_unprivileged (numpy.ndarray) : actual un privileged class
        preds_privileged (numpy.ndarray) : predicted output of privileged class
        preds_unprivileged (numpy.ndarray) : predicted output of unprivileged class
    Returns:
        demographic_parity_difference (float) : Demographic Parity Difference
    """
    cm_unprivileged = confusion_matrix(y_unprivileged, preds_unprivileged)
    cm_privileged = confusion_matrix(y_privileged, preds_privileged)

    unprivileged_PR = (cm_unprivileged[1, 1] +
                       cm_unprivileged[0, 1]) / cm_unprivileged.sum()
    privileged_PR = (cm_privileged[1, 1] +
                     cm_privileged[0, 1]) / cm_privileged.sum()
    demographic_parity_difference = unprivileged_PR - privileged_PR
    return demographic_parity_difference


def get_equal_opportunity_diff(y_privileged, y_unprivileged,
                               preds_privileged, preds_unprivileged):
    """
    Computed as the difference between true positive rate (true positives / positives)
    between the unprivileged and the privileged groups.
    Args:
        y_privileged (numpy.ndarray) : actual privileged class
        y_unprivileged (numpy.ndarray) : actual un privileged class
        preds_privileged (numpy.ndarray) : predicted output of privileged class
        preds_unprivileged (numpy.ndarray) : predicted output of unprivileged class
    Returns:
        equal_opportunity_difference (float) : equal opportunity difference
    """
    cm_unprivileged = confusion_matrix(y_unprivileged, preds_unprivileged)
    cm_privileged = confusion_matrix(y_privileged, preds_privileged)
    unprivileged_TPR = cm_unprivileged[1, 1] / cm_unprivileged[1].sum()
    privileged_TPR = cm_privileged[1, 1] / cm_privileged[1].sum()
    equal_opportunity_difference = unprivileged_TPR - privileged_TPR

    return equal_opportunity_difference


def get_equalised_odds(y_privileged, y_unprivileged,
                       preds_privileged, preds_unprivileged):
    """
    Computed as average of absolute difference between false positive rate and true positive rate
    for unprivileged and privileged groups.
    Args:
        y_privileged (numpy.ndarray) : actual privileged class
        y_unprivileged (numpy.ndarray) : actual un privileged class
        preds_privileged (numpy.ndarray) : predicted output of privileged class
        preds_unprivileged (numpy.ndarray) : predicted output of unprivileged class
    Returns:
        average_abs_odds_difference (float): average absolute equalised odds
    """

    cm_unprivileged = confusion_matrix(y_unprivileged, preds_unprivileged)
    cm_privileged = confusion_matrix(y_privileged, preds_privileged)
    unprivileged_TPR = cm_unprivileged[1, 1] / cm_unprivileged[1].sum()
    privileged_TPR = cm_privileged[1, 1] / cm_privileged[1].sum()
    unprivileged_FPR = cm_unprivileged[0, 1] / cm_unprivileged[0].sum()
    privileged_FPR = cm_privileged[0, 1] / cm_privileged[0].sum()

    # compute Equalized odds
    average_abs_odds_difference = 0.5 * \
        (abs(unprivileged_FPR - privileged_FPR) +
         abs(unprivileged_TPR - privileged_TPR))

    return average_abs_odds_difference
