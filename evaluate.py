"""Compare recommendation lists. Every list is a ranked list of title ids for one member."""
import math


def rank_of(answer, titles):
    """0-based position of the answer in the list, or None."""
    return titles.index(answer) if answer in titles else None


def score(lists, answers):
    """Return hits per member (in order of the answers), hit@k and NDCG@k for lists of length k."""
    ranks = [rank_of(answers[m], lists[m]) for m in answers]
    hits = [r is not None for r in ranks]
    ndcg = sum(1 / math.log2(r + 2) for r in ranks if r is not None) / len(ranks)
    return {"hits": hits, "hit": sum(hits) / len(hits), "ndcg": ndcg}


def mcnemar(a, b):
    """Paired exact two-sided test. Return (only a hit, only b hit, p value)."""
    only_a = sum(x and not y for x, y in zip(a, b))
    only_b = sum(y and not x for x, y in zip(a, b))
    n, low = only_a + only_b, min(only_a, only_b)
    if n == 0:
        return 0, 0, 1.0
    return only_a, only_b, min(1.0, 2 * sum(math.comb(n, i) for i in range(low + 1)) / 2 ** n)
