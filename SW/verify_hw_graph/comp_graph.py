import json
import numpy as np

# Function to load JSON data from a file
def load_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

# Function to compare two JSON datasets
def compare_edges(file1, file2):
    # Load JSON data
    data1 = load_json(file1)
    data2 = load_json(file2)

    # Check if the number of events matches
    if len(data1) != len(data2):
        print(f"\n[ERROR] Number of events mismatch:\n  File1: {len(data1)} events\n  File2: {len(data2)} events")
        return

    total_edges_file1 = 0
    total_edges_file2 = 0
    matching_edges = 0
    mismatched_edges = 0
    matching_averages = 0
    mismatched_averages = 0

    print("\n--- Comparison Start ---")

    # Compare events and edges
    for i, (event1, event2) in enumerate(zip(data1, data2)):
        avg_t1 = event1.get("average_edge", {}).get("avg_t", 0)
        avg_f1 = event1.get("average_edge", {}).get("avg_f", 0)
        avg_t2 = event2.get("average_edge", {}).get("avg_t", 0)
        avg_f2 = event2.get("average_edge", {}).get("avg_f", 0)
        pos_item1 = event1.get("pos_item", "N/A")
        pos_item2 = event2.get("pos_item", "N/A")

        # print(f"\n=== Event {i + 1} ===")
        # print(f"  File1 - pos_item: {pos_item1}, avg_t: {avg_t1}, avg_f: {avg_f1}")
        # print(f"  File2 - pos_item: {pos_item2}, avg_t: {avg_t2}, avg_f: {avg_f2}")

        # Compare pos_item
        if pos_item1 != pos_item2:
            
            print(f"[ERROR] pos_item mismatch:\n  File1: {pos_item1}\n  File2: {pos_item2}")
            continue  # Skip comparison for this event if pos_item mismatches

        # Compare average_edge
        avg_edge1 = event1.get("average_edge", {"avg_t": 0, "avg_f": 0})
        avg_edge2 = event2.get("average_edge", {"avg_t": 0, "avg_f": 0})

        # Round averages for comparison
        avg_edge1_rounded = {"avg_t": np.floor(avg_edge1["avg_t"]), "avg_f": np.floor(avg_edge1["avg_f"])}
        avg_edge2_rounded = {"avg_t": np.floor(avg_edge2["avg_t"]), "avg_f": np.floor(avg_edge2["avg_f"])}

        if avg_edge1_rounded == avg_edge2_rounded:
            matching_averages += 1
            # print("[OK] average_edge values match.")
        else:
            mismatched_averages += 1
            print(f"[ERROR] average_edge mismatch:")
            print(f"  File1: {avg_edge1_rounded}")
            print(f"  File2: {avg_edge2_rounded}")

        # Compare edges count
        edges1 = event1["edges"]
        edges2 = event2["edges"]
        total_edges_file1 += len(edges1)
        total_edges_file2 += len(edges2)

        # if len(edges1) != len(edges2):
            # print(f"[WARNING] Edges count mismatch:\n  File1: {len(edges1)} edges\n  File2: {len(edges2)} edges")

        # Compare edge details
        edges1_set = {frozenset(edge.items()) for edge in edges1}
        edges2_set = {frozenset(edge.items()) for edge in edges2}
        common_edges = edges1_set.intersection(edges2_set)
        mismatches = edges1_set.symmetric_difference(edges2_set)

        matching_edges += len(common_edges)
        mismatched_edges += len(mismatches)

        if mismatches:
            print(f"[ERROR] Mismatched edges found:")
            print(f"\n=== Event {i + 1} ===")
            print(f"  File1 - pos_item: {pos_item1}, avg_t: {avg_t1}, avg_f: {avg_f1}")
            print(f"  File2 - pos_item: {pos_item2}, avg_t: {avg_t2}, avg_f: {avg_f2}")
            for mismatch in mismatches:
                mismatch_dict = dict(mismatch)
                if mismatch_dict not in edges1 and mismatch_dict in edges2:
                    print(f"  Extra edge in File2: {mismatch_dict}")
                elif mismatch_dict in edges1 and mismatch_dict not in edges2:
                    print(f"  Missing edge in File2: {mismatch_dict}")
                else:
                    print(f"  Edge mismatch due to unknown reason: {mismatch_dict}")
        # else:
            # print("[OK] All edges match.")

    # Summary of comparison
    print("\n--- Comparison Summary ---")
    print(f"Total edges in File1: {total_edges_file1}")
    print(f"Total edges in File2: {total_edges_file2}")
    print(f"Matching edges: {matching_edges}")
    print(f"Mismatched edges: {mismatched_edges}")
    print(f"Matching average_edge values: {matching_averages}")
    print(f"Mismatched average_edge values: {mismatched_averages}")
    print("\n--- End of Comparison ---")

# Paths to the two JSON files
file1_path = "/bignobackup/hn280727/event-audio/hw_input/28-11-2024_14-03-38/formatted_edges_data copy.json"
file2_path = "/bignobackup/hn280727/event-audio/hw_input/28-11-2024_14-03-38/output.json"

# Run the comparison
compare_edges(file1_path, file2_path)
