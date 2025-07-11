def main(argv):
    if len(argv) != 3:
        print("Usage: python generate_output_files.py (ClusterID) (Number of Jobs)")
        return

    cluster_id = argv[1]
    num_jobs = int(argv[2])
    for  job_id in range(num_jobs):
        with open(f"logs/{cluster_id}.{job_id}.out", "w") as f:
            f.write(f"Cluster ID: {cluster_id}\n")
            f.write(f"Job ID: {job_id}\n")
            f.write("This is a placeholder for the output of the job.\n")
            f.write("You can add more information here as needed.\n")
        with open(f"logs/{cluster_id}.{job_id}.err", "w") as f:
            f.write(f"Cluster ID: {cluster_id}\n")
            f.write(f"Job ID: {job_id}\n")
            f.write("This is a placeholder for the error output of the job.\n")
            f.write("You can add more information here as needed.\n")

if __name__ == "__main__":
    import sys
    main(sys.argv)