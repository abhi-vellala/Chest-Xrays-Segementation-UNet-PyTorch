import pandas as pd

def check_datatable_sanity(frame_path):

    df = pd.read_excel(frame_path)

    for idx, row in df.iterrows():
        assert row['images'].split("/")[-1][:-4] == row['masks_right_lung'].split("/")[-1][:-4] == row['masks_left_lung'].split("/")[-1][:-4]

    print("Data table is good")

    return

def check_image_mask_dim(datasets):
    check = {'train': 0, 'valid': 0}
    for phase in ['train', 'valid']:
        print(f"Initialized sanity check for {phase} dataset")
        error = 0
        for idx, (image, mask) in enumerate(datasets[phase]):
            try:
                assert image.permute(0, 1, 2)[-1, :, :].size() == mask.size()
                check[phase] += 1
            except AssertionError as msg:
                # print()
                print(f"AssertionError: Size of image and mask didn't match at index: {idx}")
                error += 1
        print(f"Image and Masks on {phase} data with {error} errors")

    if check['train'] == len(datasets['train']) and check['valid'] == len(datasets['valid']):
        print("Dimensions of the dataset looks correct! Good to proceed")

    return






if __name__ == "__main__":
    print("Initialized Sanity Checks")
    print("Checking the correctness in data paths in the data table")
    frame_path = "../data/images_masks_path.xlsx"
    check_datatable_sanity(frame_path)
