def text_stats(series):
    texts = series.dropna().astype(str)

    return {
        "rows": len(texts),
        "avg_chars": texts.str.len().mean(),
        "total_words": texts.str.split().str.len().sum(),
    }

