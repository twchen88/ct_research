@dataclass
class ZeroPrescriptor:
    """Always prescribes 0 for each intervention (one per domain)."""
    n_interventions: int
    action_names: Optional[List[str]] = field(default=None)

    @classmethod
    def from_cao_mapping(cls, mapping: Dict[str, List[str]]) -> "ZeroPrescriptor":
        acts = list(mapping["actions"])
        return cls(n_interventions=len(acts), action_names=acts)

    def prescribe(self, context_df: pd.DataFrame) -> pd.DataFrame:
        n = len(context_df)
        zeros = np.zeros((n, self.n_interventions), dtype=int)
        return pd.DataFrame(zeros, columns=self.action_names)