import os, PIL
import numpy as np
import torch

all_dicom_ids = {
    ### Note: D and Y are the same case ###
    "A": "ec72dd86-36c802f0-20a909ca-8cbcc950-58733cd5",
    "B": "683aad2e-41beb1a8-9d1872c4-125b275a-ef29a7d6",
    "C": "4d994f76-a7de771a-cf65cd0f-c1250201-f04a9626",
    "D": "4288be3d-ae1b69d3-0be85637-a5236d5b-be4ac4af",
    "E": "05dd9e22-b8766b89-49e21ff4-9cad2776-908d4c9f",
    "F": "e6114f27-d0cdef4a-13ef26b5-86198fa7-0890f0be",
    "G": "a04250da-25655fa3-7b75e707-a862738a-375e4e9f",
    "H": "dffe7130-cb1ee280-5aee23a9-754b56f9-b20c2a3c",
    "I": "70e31905-dd605e80-305f056b-4f88ec80-cbb4b3fb",
    "J": "6d7c4296-157fc904-b6d6c70d-f3c54944-61ba51e2",
    "K": "82043e9e-dc650774-246cc5bd-efb75286-bebc4801",
    "L": "53588137-69f5216f-022b1177-b7ed05ef-9a670fd8",
    "M": "90bba64d-6913b983-a1740cf9-5fb87f7e-9171482d",
    # "N": "919158fb-4f0d9b66-46719ab6-5d584449-1a3ad8de", # No.
    "O": "ce6c73a2-bfbdbdf8-f7f014a2-bfffc5e3-232d2d80",
    "P": "425d59af-b3a07390-48699ce4-edd9cf7d-3b4faafe",
    "Q": "a04250da-25655fa3-7b75e707-a862738a-375e4e9f",
    "R": "cd27ccb8-f83311dd-ceacc3b7-501d40c2-6f1071f8",
    "S": "4a04164c-bf7a47b2-39273bf3-6f841e34-278431eb",
    "T": "467886fc-bdd148bc-96415ce2-3ea24428-0ee1d9a1",
    "U": "315a2ff9-d2cc7585-47e3c881-524b9634-158b6ae8",
    "V": "1201d2ae-1e36ac1e-037a7be9-82d08c96-044333da",
    "W": "919158fb-4f0d9b66-46719ab6-5d584449-1a3ad8de",
    "X": "f24dcfb8-8d336748-8d0d5686-a52f7cc9-2aefd3a6",
    "Y": "4288be3d-ae1b69d3-0be85637-a5236d5b-be4ac4af",
}

def get_bouding_box(annotated_img_path, orginial_img_path):

    annotated_img = PIL.Image.open(annotated_img_path).convert("RGB")
    original_img = PIL.Image.open(orginial_img_path).convert("RGB")

    annotated_img_arr = np.array(annotated_img)
    a_y, a_x = np.where(
        (
            (annotated_img_arr[:, :, 0] != annotated_img_arr[:, :, 1])
            | (annotated_img_arr[:, :, 0] != annotated_img_arr[:, :, 2])
        )
    )

    a_x_max = a_x.max()
    a_x_min = a_x.min()
    a_y_max = a_y.max()
    a_y_min = a_y.min()

    a_w, a_h = annotated_img.size
    o_w, o_h = original_img.size

    o_x_max = a_x_max * (o_w / a_w)
    o_x_min = a_x_min * (o_w / a_w)
    o_y_max = a_y_max * (o_h / a_h)
    o_y_min = a_y_min * (o_h / a_h)

    return np.array([o_x_min, o_y_min, o_x_max, o_y_max])


def get_anns(folder_path, dataset): #"radiologists_annotated"
    radiologists_anns = []
    for k, d_id in all_dicom_ids.items():
        bboxes = []
        image_path = dataset.get_image_path_from_dicom_id(d_id)
        for file in os.listdir(os.path.join(folder_path, f"{k}_{d_id}")):
            if file.endswith(".png"):
                # print(file)
                disease = file.split("-")[0]
                bbox = get_bouding_box(
                    os.path.join(folder_path, f"{k}_{d_id}", file),
                    image_path,
                ).astype(int)

                bboxes.append(
                    {
                        "label": dataset.disease_to_idx(disease),
                        "box": bbox,
                    }
                )

        radiologists_anns.append({
            "labels":torch.tensor([b['label'] for b in  bboxes]),
            "boxes": torch.tensor([b['box'] for b in bboxes]),
            "image_path": image_path,
            "dicom_id": d_id,
            "encoding": k
        })
    return radiologists_anns

