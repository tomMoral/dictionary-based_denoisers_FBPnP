import os
import subprocess


def download_bsds500(dest_folder="."):
    # Extract the repo name from the URL
    repo_url = "https://github.com/BIDS/BSDS500.git"
    repo_name = repo_url.rstrip("/").split("/")[-1].replace(".git", "")
    repo_path = os.path.join(dest_folder, repo_name)

    # Check if the repo folder already exists
    if not os.path.exists(repo_path):
        print(f"Repository not found. Cloning {repo_name} into {dest_folder}...")
        try:
            subprocess.run(["git", "clone", repo_url, repo_path], check=True)
            print(f"Successfully cloned {repo_name}!")
        except subprocess.CalledProcessError as e:
            print(f"Error occurred while cloning the repository: {e}")
    else:
        print(f"Repository {repo_name} already exists in {dest_folder}.")
    return repo_path


if __name__ == "__main__":
    download_bsds500()
