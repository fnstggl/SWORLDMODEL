"""Tests for demographic -> value-variable mapping and the individual-opinion prediction primitives."""
from swm.variables.demographic_values import VALUE_DIMS, demographic_to_values, value_vector


def test_value_dims_present_and_in_range():
    demo = {"religion": "atheist", "attendance": "never", "ideology": "very liberal", "party": "democrat",
            "age": "18-29", "education": "postgraduate", "income": "$100,000 or more"}
    v = demographic_to_values(demo)
    assert set(v) == set(VALUE_DIMS)
    for d in VALUE_DIMS:
        lo = -1.0 if d == "economic_left" else 0.0
        assert lo <= v[d] <= 1.0


def test_religiosity_ordering_matches_intuition():
    devout = {"religion": "protestant", "attendance": "more than once a week"}
    secular = {"religion": "atheist", "attendance": "never"}
    assert demographic_to_values(devout)["religiosity"] > demographic_to_values(secular)["religiosity"]


def test_ideology_moves_the_right_axes():
    left = {"ideology": "very liberal", "party": "democrat"}
    right = {"ideology": "very conservative", "party": "republican"}
    vl, vr = demographic_to_values(left), demographic_to_values(right)
    assert vr["traditionalism"] > vl["traditionalism"]
    assert vr["social_progressive"] < vl["social_progressive"]
    assert vl["economic_left"] > vr["economic_left"]        # left leans pro-redistribution


def test_value_vector_length_and_unknown_is_neutral():
    assert len(value_vector({})) == len(VALUE_DIMS)          # empty demo -> neutral profile, no crash
    v = demographic_to_values({})
    assert all(-1.0 <= v[d] <= 1.0 for d in VALUE_DIMS)
