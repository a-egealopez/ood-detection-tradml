from teaching.synthetic_2d_datasets import SyntheticDatasetGenerator


def test_all_datasets_generated_with_shapes_and_contamination():
    gen = SyntheticDatasetGenerator()
    for key in ["blobs", "moons", "circles", "swiss_roll"]:
        X, y = gen.generate(key, n_samples=300, contamination=0.1)
        assert X.shape == (300, 2)
        assert len(y) == 300
        assert int(y.sum()) == 30  # 10% contamination


def test_invalid_dataset_raises():
    try:
        SyntheticDatasetGenerator.generate("not_a_dataset")
    except ValueError:
        return
    raise AssertionError("unknown dataset name should raise ValueError")
