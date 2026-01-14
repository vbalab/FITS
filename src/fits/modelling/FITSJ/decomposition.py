...


@torch.no_grad()
def EvaluateFITSJWithDecomposition(
    model: ForecastingModel,
    test_loader: DataLoader,
    normalization: NormalizationStats,
    nsample: int = 5,
    folder_name: str | None = None,
): ...

def PlotFITSJDecomposition(
    eval_foldername: str | Path,
    nsample: int = 10,
    sample_index: int = 0,
    feature_index: int | Sequence[int] | None = None,
    ...,
    figsize: tuple[int, int] = (8, 6),
) -> None: ...
