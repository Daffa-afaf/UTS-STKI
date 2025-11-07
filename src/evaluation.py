def precision_at_k(retrieved, relevant, k):
    """Calculate precision at k"""
    if not retrieved:
        return 0
    retrieved_at_k = retrieved[:k]
    relevant_at_k = [doc for doc in retrieved_at_k if doc in relevant]
    return len(relevant_at_k) / k if k > 0 else 0


def recall_at_k(retrieved, relevant, k):
    """Calculate recall at k"""
    if not retrieved or not relevant:
        return 0
    retrieved_at_k = retrieved[:k]
    relevant_at_k = [doc for doc in retrieved_at_k if doc in relevant]
    return len(relevant_at_k) / len(relevant) if len(relevant) > 0 else 0


def average_precision(retrieved, relevant):
    """Calculate Average Precision (AP)"""
    if not relevant:
        return 0
    ap = 0
    num_relevant = 0
    for i, doc in enumerate(retrieved):
        if doc in relevant:
            num_relevant += 1
            ap += num_relevant / (i + 1)
    return ap / len(relevant)


def mean_average_precision(results, ground_truth):
    """Calculate Mean Average Precision (MAP@k)"""
    aps = []
    for query in results:
        # pastikan key 'MAP@k' tidak ikut dihitung ulang
        if query == 'MAP@k':
            continue
        retrieved = results[query]['retrieved_docs']
        relevant = ground_truth.get(query, [])
        ap = average_precision(retrieved, relevant)
        aps.append(ap)
    return sum(aps) / len(aps) if aps else 0


def evaluate_model(model, queries, ground_truth, k=10):
    """
    Evaluate the model for given queries and ground truth.
    Returns dict with precision, recall, AP for each query, and MAP@k
    """
    results = {}

    for query in queries:
        retrieved_with_scores = model.search(query)

        #  Tambahan keamanan: pastikan hasil berupa list
        if not isinstance(retrieved_with_scores, list):
            print(f"[WARNING] Model '{type(model).__name__}' tidak mengembalikan list untuk query '{query}'")
            retrieved_with_scores = []

        #  Jika hasil pencarian kosong
        if not retrieved_with_scores:
            print(f"[INFO] Tidak ada hasil untuk query: '{query}'")
            results[query] = {
                'precision': 0,
                'recall': 0,
                'ap': 0,
                'retrieved': [],
                'retrieved_docs': []
            }
            continue

        #  Deteksi struktur hasil pencarian (tuple 3, tuple 2, atau string)
        first_item = retrieved_with_scores[0]
        if isinstance(first_item, tuple):
            if len(first_item) == 3:
                retrieved = [doc for doc, _, _ in retrieved_with_scores]
            elif len(first_item) == 2:
                retrieved = [doc for doc, _ in retrieved_with_scores]
            else:
                retrieved = [doc for doc, *_ in retrieved_with_scores]
        else:
            # kalau bukan tuple (misalnya list of doc_id)
            retrieved = retrieved_with_scores

        relevant = ground_truth.get(query, [])

        precision = precision_at_k(retrieved, relevant, k)
        recall = recall_at_k(retrieved, relevant, k)
        ap = average_precision(retrieved, relevant)

        results[query] = {
            'precision': precision,
            'recall': recall,
            'ap': ap,
            'retrieved': retrieved_with_scores,
            'retrieved_docs': retrieved
        }

    #  Hitung MAP@k (tanpa menghitung key 'MAP@k' itu sendiri)
    results['MAP@k'] = mean_average_precision(results, ground_truth)

    return results
