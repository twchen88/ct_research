class UnileafPrescriptor(EspEvaluator):
    """
    An Unileaf Prescriptor makes prescriptions given an ESP candidate and a context DataFrame.
    It is also an EspEvaluator implementation that returns metrics for ESP candidates.
    """

    def __init__(self,
                 config: Dict[str, Any],
                 evaluation_df: pd.DataFrame,
                 data_encoder: DataEncoder,
                 predictors: List[Predictor]):
        """
        Constructs a prescriptor evaluator
        :param config: the ESP experiment config dictionary
        :param evaluation_df: the Pandas DataFrame to use to evaluate the candidates
        :param data_encoder: the DataEncoder used to encode the dataset
        :param predictors: the predictors this prescriptor relies on
        """
        # Instantiate EspEvaluator
        # Note: sets self.config
        super().__init__(config)

        # CAO
        self.cao_mapping = {"context": self.get_context_field_names(config),
                            "actions": self.get_action_field_names(config),
                            "outcomes": self.get_fitness_metrics(config)}
        self.context_df = evaluation_df[self.cao_mapping["context"]]
        self.row_index = self.context_df.index

        # Convert the context DataFrame to a format a NN can ingest
        self.context_as_nn_input = self.convert_to_nn_input(self.context_df)

        # Data encoder
        self.data_encoder = data_encoder

        # Predictors
        self.predictors = predictors

    @staticmethod
    def convert_to_nn_input(context_df: pd.DataFrame) -> List[np.ndarray]:
        """
        Converts a context DataFrame to a list of numpy arrays a neural network can ingest
        :param context_df: a DataFrame containing inputs for a neural network. Number of inputs and size must match
        :return: a list of numpy ndarray, on ndarray per neural network input
        """
        # The NN expects a list of i inputs by s samples (e.g. 9 x 299).
        # So convert the data frame to a numpy array (gives shape 299 x 9), transpose it (gives 9 x 299)
        # and convert to list(list of 9 arrays of 299)
        context_as_nn_input = list(context_df.to_numpy().transpose())
        # Convert each column's list of 1D array to a 2D array
        context_as_nn_input = [np.stack(context_as_nn_input[i], axis=0) for i in
                               range(len(context_as_nn_input))]
        return context_as_nn_input

    def evaluate_candidate(self, candidate):
        """
        Evaluates a single Prescriptor candidate and returns its metrics.
        Implements the EspEvaluator interface
        :param candidate: a Keras neural network or rule based Prescriptor candidate
        :return metrics: A dictionary of {'metric_name': metric_value}
        """
        # Prescribe actions
        prescribed_actions_df = self.prescribe(candidate)

        # Aggregate the context and actions dataframes.
        context_actions_df = pd.concat([self.context_df,
                                        prescribed_actions_df],
                                       axis=1)

        # Compute the metrics
        metrics = self._compute_metrics(context_actions_df)
        return metrics

    def _compute_metrics(self, context_actions_df):
        """
        Computes metrics from the passed context/actions DataFrame using the instance's trained predictors.
        :param context_actions_df: a DataFrame of context / prescribed actions
        :return: A dictionary of {'metric_name': metric_value}
        """
        # Get the predicted outcomes from the predictors
        metrics = {}
        for predictor in self.predictors:
            predicted_outcomes = predictor.predict(context_actions_df)

            # UN-853: Decode predictions before computing numerical metrics, if a data_encoder is available
            if self.data_encoder is not None:
                decoded_predicted_outcomes = self.data_encoder.decode_as_df(predicted_outcomes)
            else:
                decoded_predicted_outcomes = predicted_outcomes

            # Only add a metric for the outcomes the prescriptor is interested in
            for outcome in self.cao_mapping["outcomes"]:
                # Add the metrics that have been produced by this predictor
                if outcome in predictor.cao_mapping["outcomes"]:
                    # Check the type of metric: numerical or categorical?
                    if decoded_predicted_outcomes[[outcome]].iloc[:, 0].dtype == object:
                        # Categorical outcome. Use the *encoded* predicted outcome.
                        preds = predicted_outcomes[outcome]
                        # Classifiers return the category's index in the list of categories, so we can take the mean
                        # of the encoded outcomes. Note: this works because Outcomes are encoded using LabelEncoder
                        # AND the user defined order for each Outcome categories.
                        metrics[outcome] = preds.mean()
                    else:
                        # UN-853: Numerical outcome. Use the *decoded*, i.e. scaled back, predicted outcome
                        preds = decoded_predicted_outcomes[outcome]
                        # Regressors produce floats: take the mean of the decoded outcome
                        metrics[outcome] = preds.mean()
        return metrics

    def prescribe(self, candidate, context_df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Generates prescriptions using the passed candidate and context
        :param candidate: an ESP candidate, either neural network or rules
        :param context_df: a DataFrame containing the context to prescribe for,
         or None to use the instance one
        :return: a DataFrame containing actions prescribed for each context
        """
        if context_df is None:
            # No context is provided, use the instance's one
            context_as_nn_input = self.context_as_nn_input
            row_index = self.row_index
        else:
            # Convert the context DataFrame to something more suitable for neural networks
            context_as_nn_input = self.convert_to_nn_input(context_df)
            # Use the context's row index
            row_index = context_df.index

        is_rule_based = isinstance(candidate, RuleSet)
        if is_rule_based:
            actions = self._prescribe_from_rules(candidate, context_as_nn_input)
        else:
            actions = self._prescribe_from_nn(candidate, context_as_nn_input)

        # Convert the prescribed actions to a DataFrame
        prescribed_actions_df = pd.DataFrame(actions,
                                             columns=self.cao_mapping["actions"],
                                             index=row_index)
        # UN-2430 Decode the softmaxes, if any, back into categories
        prescribed_actions_df = self.data_encoder.decode_as_df(prescribed_actions_df)
        # UN0-240 Re-encode the actions into what the predictors expect (e.g. one-hots for categorical data)
        prescribed_actions_df = self.data_encoder.encode_as_df(prescribed_actions_df)
        return prescribed_actions_df

    def _prescribe_from_rules(self, candidate, context_as_nn_input: List[np.ndarray]):
        """
        Generates prescriptions using the passed rules model candidate and context
        :param candidate: a rules model candidate
        :param context_as_nn_input: a numpy array containing the context to prescribe for
        :return: a dictionary of action name to list of action values
        """
        cand_states = RuleSetConfigHelper.get_states(self.config)
        cand_actions = RuleSetConfigHelper.get_actions(self.config)
        candidate = RuleSetBinding(candidate, cand_states, cand_actions)
        rules_encoder = RulesDataEncoder(candidate.actions)
        evaluator = RuleSetBindingEvaluator()
        rules_input = rules_encoder.encode_to_rules_data(context_as_nn_input)
        rules_output = evaluator.evaluate(candidate, rules_input)
        actions = rules_encoder.decode_from_rules_data(rules_output)
        return actions

    def _prescribe_from_nn(self, candidate, context_as_nn_input: List[np.ndarray]) -> Dict[str, Any]:
        """
        Generates prescriptions using the passed neural network candidate and context
        :param candidate: a Keras neural network candidate
        :param context_as_nn_input: a numpy array containing the context to prescribe for
        :return: a dictionary of action name to action value or list of action values
        """
        # Get the prescribed actions
        prescribed_actions = candidate.predict(context_as_nn_input)
        actions = {}

        if self._is_single_action_prescriptor():
            # Put the single action in an array to process it like multiple actions
            prescribed_actions = [prescribed_actions]

        for index, action_col in enumerate(self.cao_mapping["actions"]):
            if self._is_scalar(prescribed_actions[index]):
                # We have a single row and this action is numerical. Convert it to a scalar.
                actions[action_col] = prescribed_actions[index].item()
            else:
                actions[action_col] = prescribed_actions[index].tolist()
        return actions

    def _is_single_action_prescriptor(self):
        """
        Checks how many Actions have been defined in the Context, Actions, Outcomes mapping.
        :return: True if only 1 action is defined, False otherwise
        """
        return len(self.cao_mapping["actions"]) == 1

    @staticmethod
    def _is_scalar(prescribed_action):
        """
        Checks if the prescribed action contains a single value, i.e. a scalar, or an array.
        A prescribed action contains a single value if it has been prescribed for a single context sample
        :param prescribed_action: a scalar or an array
        :return: True if the prescribed action contains a scalar, False otherwise.
        """
        return prescribed_action.shape[0] == 1 and prescribed_action.shape[1] == 1

    @staticmethod
    def get_context_field_names(config: Dict[str, Any]) -> List[str]:
        """
        Returns the list of Context column names
        :param config: the ESP experiment config dictionary
        :return: the list of Context column names
        """
        nn_inputs = config["network"]["inputs"]
        contexts = [nn_input["name"] for nn_input in nn_inputs]
        return contexts

    @staticmethod
    def get_action_field_names(config: Dict[str, Any]) -> List[str]:
        """
        Returns the list of Action column names
        :param config: the ESP experiment config dictionary
        :return: the list of Action column names
        """
        nn_outputs = config["network"]["outputs"]
        actions = [nn_output["name"] for nn_output in nn_outputs]
        return actions

    @staticmethod
    def get_fitness_metrics(config: Dict[str, Any]) -> List[str]:
        """
        Returns the list of fitness metric names (Outcomes) to optimize.
        :param config: the ESP experiment config dictionary
        :return: the list of fitness metric names
        """
        metrics = config["evolution"]["fitness"]
        fitness_metrics = [metric["metric_name"] for metric in metrics]
        return fitness_metrics
