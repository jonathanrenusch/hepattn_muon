#!/usr/bin/env python3
"""Simple verification of the changes."""

# Test 1: Check that the global constant is accessible
try:
    import sys
    sys.path.append('/shared/tracking/hepattn_muon/src')
    from hepattn.experiments.atlas_muon.evaluate_hit_filter_dataloader import DEFAULT_WORKING_POINTS
    print("✅ Global working points accessible:", DEFAULT_WORKING_POINTS)
    print(f"✅ Contains {len(DEFAULT_WORKING_POINTS)} working points")
    expected = [0.96, 0.965, 0.97, 0.975, 0.98, 0.985, 0.99, 0.995]
    if DEFAULT_WORKING_POINTS == expected:
        print("✅ Working points match expected values")
    else:
        print("❌ Working points don't match expected values")
        print(f"Expected: {expected}")
        print(f"Got: {DEFAULT_WORKING_POINTS}")
except Exception as e:
    print(f"❌ Error importing global working points: {e}")

# Test 2: Check function signatures
try:
    from hepattn.experiments.atlas_muon.evaluate_hit_filter_dataloader import AtlasMuonEvaluatorDataLoader
    import inspect
    
    # Check plot_efficiency_vs_pt signature
    sig = inspect.signature(AtlasMuonEvaluatorDataLoader.plot_efficiency_vs_pt)
    working_points_param = sig.parameters.get('working_points')
    if working_points_param and working_points_param.default is None:
        print("✅ plot_efficiency_vs_pt uses default working points")
    else:
        print(f"❌ plot_efficiency_vs_pt parameter: {working_points_param}")
    
    # Check plot_working_point_performance signature  
    sig = inspect.signature(AtlasMuonEvaluatorDataLoader.plot_working_point_performance)
    working_points_param = sig.parameters.get('working_points')
    if working_points_param and working_points_param.default is None:
        print("✅ plot_working_point_performance uses default working points")
    else:
        print(f"❌ plot_working_point_performance parameter: {working_points_param}")
        
except Exception as e:
    print(f"❌ Error checking function signatures: {e}")

print("\n🎉 Verification complete!")
