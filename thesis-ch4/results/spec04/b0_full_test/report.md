# SPEC-04-A2 全 test B0

{
  "spec": "SPEC-04-A2",
  "n_test": 932,
  "n_generated": 932,
  "n_interrupted": 0,
  "interrupted": [],
  "complete": true,
  "overall": {
    "sample_count": 932,
    "expected_count": 932,
    "complete": true,
    "layer_a_pass_count": 355,
    "layer_a_fail_count": 577,
    "parse_fail_rate": 0.619098712446352,
    "format_ok_rate": 0.0,
    "layer_a_reason_histogram": {
      "no_interval": 543,
      "single_point": 30,
      "multi_interval": 0,
      "missing_unit": 3,
      "ambiguous_clock": 1
    },
    "layer_b_reason_histogram": {
      "valid": 355,
      "reverse_order": 0,
      "zero_duration": 0
    },
    "out_of_range_count": 2,
    "duration_distribution_sec": {
      "count": 355,
      "min": 0.00999999999999801,
      "median": 4.84,
      "max": 14.999999999999996,
      "quantiles": {
        "0.05": 0.5109999999999972,
        "0.25": 2.360000000000001,
        "0.5": 4.84,
        "0.75": 4.959999999999999,
        "0.95": 9.84
      }
    },
    "start_distribution_sec": {
      "count": 355,
      "min": 1.04,
      "median": 8.32,
      "max": 42.2,
      "quantiles": {
        "0.05": 2.428,
        "0.25": 6.08,
        "0.5": 8.32,
        "0.75": 16.96,
        "0.95": 22.4
      }
    },
    "reverse_order_intervals": [],
    "avg_frames": 12,
    "full_clip_ratio": 0.0,
    "truncation_suspected_count": 0,
    "rate_denominator": "persisted records only; interrupted attempts excluded",
    "distribution_population": "Layer A passed and Layer B valid/zero_duration; reverse_order listed separately",
    "stop_reason_histogram": {
      "eos": 829,
      "max_new_tokens": 103
    },
    "truncated_count": 103,
    "truncated_rate": 0.11051502145922747,
    "evidence_malformed_count": 0
  },
  "by_category": {
    "二轮车辆闯入": {
      "sample_count": 221,
      "expected_count": 221,
      "complete": true,
      "layer_a_pass_count": 128,
      "layer_a_fail_count": 93,
      "parse_fail_rate": 0.420814479638009,
      "format_ok_rate": 0.0,
      "layer_a_reason_histogram": {
        "no_interval": 87,
        "single_point": 4,
        "multi_interval": 0,
        "missing_unit": 1,
        "ambiguous_clock": 1
      },
      "layer_b_reason_histogram": {
        "valid": 128,
        "reverse_order": 0,
        "zero_duration": 0
      },
      "out_of_range_count": 0,
      "duration_distribution_sec": {
        "count": 128,
        "min": 0.029999999999997584,
        "median": 2.3999999999999995,
        "max": 7.039999999999999,
        "quantiles": {
          "0.05": 0.7199999999999999,
          "0.25": 2.3599999999999994,
          "0.5": 2.3999999999999995,
          "0.75": 2.4399999999999995,
          "0.95": 3.9759999999999804
        }
      },
      "start_distribution_sec": {
        "count": 128,
        "min": 1.04,
        "median": 6.2,
        "max": 22.56,
        "quantiles": {
          "0.05": 2.4,
          "0.25": 6.04,
          "0.5": 6.2,
          "0.75": 8.33,
          "0.95": 20.0
        }
      },
      "reverse_order_intervals": [],
      "avg_frames": 12,
      "full_clip_ratio": 0.0,
      "truncation_suspected_count": 0,
      "rate_denominator": "persisted records only; interrupted attempts excluded",
      "distribution_population": "Layer A passed and Layer B valid/zero_duration; reverse_order listed separately",
      "stop_reason_histogram": {
        "eos": 196,
        "max_new_tokens": 25
      },
      "truncated_count": 25,
      "truncated_rate": 0.11312217194570136,
      "evidence_malformed_count": 0
    },
    "占道施工": {
      "sample_count": 223,
      "expected_count": 223,
      "complete": true,
      "layer_a_pass_count": 161,
      "layer_a_fail_count": 62,
      "parse_fail_rate": 0.27802690582959644,
      "format_ok_rate": 0.0,
      "layer_a_reason_histogram": {
        "no_interval": 60,
        "single_point": 1,
        "multi_interval": 0,
        "missing_unit": 1,
        "ambiguous_clock": 0
      },
      "layer_b_reason_histogram": {
        "valid": 161,
        "reverse_order": 0,
        "zero_duration": 0
      },
      "out_of_range_count": 1,
      "duration_distribution_sec": {
        "count": 161,
        "min": 0.00999999999999801,
        "median": 4.92,
        "max": 14.999999999999996,
        "quantiles": {
          "0.05": 1.24,
          "0.25": 4.84,
          "0.5": 4.92,
          "0.75": 4.960000000000001,
          "0.95": 9.920000000000002
        }
      },
      "start_distribution_sec": {
        "count": 161,
        "min": 1.64,
        "median": 12.08,
        "max": 42.2,
        "quantiles": {
          "0.05": 2.4,
          "0.25": 6.52,
          "0.5": 12.08,
          "0.75": 20.24,
          "0.95": 22.92
        }
      },
      "reverse_order_intervals": [],
      "avg_frames": 12,
      "full_clip_ratio": 0.0,
      "truncation_suspected_count": 0,
      "rate_denominator": "persisted records only; interrupted attempts excluded",
      "distribution_population": "Layer A passed and Layer B valid/zero_duration; reverse_order listed separately",
      "stop_reason_histogram": {
        "eos": 220,
        "max_new_tokens": 3
      },
      "truncated_count": 3,
      "truncated_rate": 0.013452914798206279,
      "evidence_malformed_count": 0
    },
    "多车事故": {
      "sample_count": 34,
      "expected_count": 34,
      "complete": true,
      "layer_a_pass_count": 5,
      "layer_a_fail_count": 29,
      "parse_fail_rate": 0.8529411764705882,
      "format_ok_rate": 0.0,
      "layer_a_reason_histogram": {
        "no_interval": 28,
        "single_point": 1,
        "multi_interval": 0,
        "missing_unit": 0,
        "ambiguous_clock": 0
      },
      "layer_b_reason_histogram": {
        "valid": 5,
        "reverse_order": 0,
        "zero_duration": 0
      },
      "out_of_range_count": 1,
      "duration_distribution_sec": {
        "count": 5,
        "min": 0.15999999999999837,
        "median": 4.920000000000002,
        "max": 14.880000000000003,
        "quantiles": {
          "0.05": 0.6479999999999988,
          "0.25": 2.6000000000000005,
          "0.5": 4.920000000000002,
          "0.75": 4.960000000000001,
          "0.95": 12.896
        }
      },
      "start_distribution_sec": {
        "count": 5,
        "min": 3.84,
        "median": 22.08,
        "max": 22.32,
        "quantiles": {
          "0.05": 6.064,
          "0.25": 14.96,
          "0.5": 22.08,
          "0.75": 22.2,
          "0.95": 22.296
        }
      },
      "reverse_order_intervals": [],
      "avg_frames": 12,
      "full_clip_ratio": 0.0,
      "truncation_suspected_count": 0,
      "rate_denominator": "persisted records only; interrupted attempts excluded",
      "distribution_population": "Layer A passed and Layer B valid/zero_duration; reverse_order listed separately",
      "stop_reason_histogram": {
        "eos": 31,
        "max_new_tokens": 3
      },
      "truncated_count": 3,
      "truncated_rate": 0.08823529411764706,
      "evidence_malformed_count": 0
    },
    "异常停车": {
      "sample_count": 210,
      "expected_count": 210,
      "complete": true,
      "layer_a_pass_count": 3,
      "layer_a_fail_count": 207,
      "parse_fail_rate": 0.9857142857142858,
      "format_ok_rate": 0.0,
      "layer_a_reason_histogram": {
        "no_interval": 204,
        "single_point": 3,
        "multi_interval": 0,
        "missing_unit": 0,
        "ambiguous_clock": 0
      },
      "layer_b_reason_histogram": {
        "valid": 3,
        "reverse_order": 0,
        "zero_duration": 0
      },
      "out_of_range_count": 0,
      "duration_distribution_sec": {
        "count": 3,
        "min": 2.5600000000000005,
        "median": 4.879999999999999,
        "max": 4.880000000000003,
        "quantiles": {
          "0.05": 2.7920000000000003,
          "0.25": 3.7199999999999998,
          "0.5": 4.879999999999999,
          "0.75": 4.880000000000001,
          "0.95": 4.880000000000003
        }
      },
      "start_distribution_sec": {
        "count": 3,
        "min": 3.8,
        "median": 21.96,
        "max": 22.08,
        "quantiles": {
          "0.05": 5.616,
          "0.25": 12.88,
          "0.5": 21.96,
          "0.75": 22.02,
          "0.95": 22.067999999999998
        }
      },
      "reverse_order_intervals": [],
      "avg_frames": 12,
      "full_clip_ratio": 0.0,
      "truncation_suspected_count": 0,
      "rate_denominator": "persisted records only; interrupted attempts excluded",
      "distribution_population": "Layer A passed and Layer B valid/zero_duration; reverse_order listed separately",
      "stop_reason_histogram": {
        "eos": 210
      },
      "truncated_count": 0,
      "truncated_rate": 0.0,
      "evidence_malformed_count": 0
    },
    "抛洒物": {
      "sample_count": 21,
      "expected_count": 21,
      "complete": true,
      "layer_a_pass_count": 1,
      "layer_a_fail_count": 20,
      "parse_fail_rate": 0.9523809523809523,
      "format_ok_rate": 0.0,
      "layer_a_reason_histogram": {
        "no_interval": 19,
        "single_point": 1,
        "multi_interval": 0,
        "missing_unit": 0,
        "ambiguous_clock": 0
      },
      "layer_b_reason_histogram": {
        "valid": 1,
        "reverse_order": 0,
        "zero_duration": 0
      },
      "out_of_range_count": 0,
      "duration_distribution_sec": {
        "count": 1,
        "min": 5.0,
        "median": 5.0,
        "max": 5.0,
        "quantiles": {
          "0.05": 5.0,
          "0.25": 5.0,
          "0.5": 5.0,
          "0.75": 5.0,
          "0.95": 5.0
        }
      },
      "start_distribution_sec": {
        "count": 1,
        "min": 2.48,
        "median": 2.48,
        "max": 2.48,
        "quantiles": {
          "0.05": 2.48,
          "0.25": 2.48,
          "0.5": 2.48,
          "0.75": 2.48,
          "0.95": 2.48
        }
      },
      "reverse_order_intervals": [],
      "avg_frames": 12,
      "full_clip_ratio": 0.0,
      "truncation_suspected_count": 0,
      "rate_denominator": "persisted records only; interrupted attempts excluded",
      "distribution_population": "Layer A passed and Layer B valid/zero_duration; reverse_order listed separately",
      "stop_reason_histogram": {
        "eos": 19,
        "max_new_tokens": 2
      },
      "truncated_count": 2,
      "truncated_rate": 0.09523809523809523,
      "evidence_malformed_count": 0
    },
    "拥堵": {
      "sample_count": 223,
      "expected_count": 223,
      "complete": true,
      "layer_a_pass_count": 57,
      "layer_a_fail_count": 166,
      "parse_fail_rate": 0.7443946188340806,
      "format_ok_rate": 0.0,
      "layer_a_reason_histogram": {
        "no_interval": 145,
        "single_point": 20,
        "multi_interval": 0,
        "missing_unit": 1,
        "ambiguous_clock": 0
      },
      "layer_b_reason_histogram": {
        "valid": 57,
        "reverse_order": 0,
        "zero_duration": 0
      },
      "out_of_range_count": 0,
      "duration_distribution_sec": {
        "count": 57,
        "min": 0.019999999999999574,
        "median": 4.92,
        "max": 14.959999999999997,
        "quantiles": {
          "0.05": 0.09200000000000125,
          "0.25": 4.84,
          "0.5": 4.92,
          "0.75": 4.999999999999998,
          "0.95": 9.895999999999999
        }
      },
      "start_distribution_sec": {
        "count": 57,
        "min": 2.4,
        "median": 16.92,
        "max": 31.64,
        "quantiles": {
          "0.05": 7.36,
          "0.25": 12.2,
          "0.5": 16.92,
          "0.75": 18.5,
          "0.95": 22.407999999999998
        }
      },
      "reverse_order_intervals": [],
      "avg_frames": 12,
      "full_clip_ratio": 0.0,
      "truncation_suspected_count": 0,
      "rate_denominator": "persisted records only; interrupted attempts excluded",
      "distribution_population": "Layer A passed and Layer B valid/zero_duration; reverse_order listed separately",
      "stop_reason_histogram": {
        "eos": 153,
        "max_new_tokens": 70
      },
      "truncated_count": 70,
      "truncated_rate": 0.31390134529147984,
      "evidence_malformed_count": 0
    }
  }
}

可复现性诊断：
{
  "expected_overlap": 312,
  "compared_count": 312,
  "exact_count": 312,
  "reproduced_exact_rate": 1.0,
  "comparison_complete": true,
  "diagnostic_only": true,
  "not_a_pass_fail_threshold": true
}

不一致文本详见 reproducibility.json；不作为验收失败门。
