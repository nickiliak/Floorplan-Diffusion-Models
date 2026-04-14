import zipfile
import os

def extract_resplan():
    # Configuration
    zip_filename = "external/ResPlan/ResPlan.zip"
    target_file = "ResPlan.pkl"
    output_dir = os.path.join("data", "raw")

    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    try:
        with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
            # Check if the file exists within the zip
            if target_file in zip_ref.namelist():
                # Extract only the specific file
                zip_ref.extract(target_file, path=output_dir)
                print(f"Successfully extracted '{target_file}' to '{output_dir}'.")
            else:
                print(f"Error: '{target_file}' not found in {zip_filename}.")
                
    except FileNotFoundError:
        print(f"Error: The file '{zip_filename}' was not found in the current directory.")
    except zipfile.BadZipFile:
        print(f"Error: '{zip_filename}' does not appear to be a valid zip file.")

if __name__ == "__main__":
    extract_resplan()