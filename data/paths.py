import os
from enum import Enum

class TabularDataPaths():
    
    class SpreadSheet():

        def get_sreadsheet(mimic_folder_path, path):
            return os.path.join(mimic_folder_path, path)

        root_path = "spreadsheets"
        cxr_meta = os.path.join(root_path, "cxr_meta.csv")
        cxr_meta_with_stay_id_only = os.path.join(
            root_path, "cxr_meta_with_stay_id_only.csv")

        class CXR_JPG():
            root_path = os.path.join("spreadsheets", "CXR-JPG")
            cxr_chexpert = os.path.join(root_path, "cxr_chexpert.csv")
            cxr_negbio = os.path.join(root_path, "cxr_negbio.csv")
            cxr_split = os.path.join(root_path, "cxr_split.csv")

        class EyeGaze():
            root_path = os.path.join("spreadsheets", "EyeGaze")
            bounding_boxes = os.path.join(root_path, "bounding_boxes.csv")
            fixations = os.path.join(root_path, "fixations.csv")
            master_sheet_with_updated_stayId = os.path.join(
                root_path, "master_sheet_with_updated_stayId.csv")

        class REFLACX():
            root_path = os.path.join("spreadsheets", "REFLACX")
            metadata = os.path.join(root_path, "metadata.csv")

    class PatientDataPaths():

        def get_patient_path(mimic_folder_path, patient_id, path):
            return os.path.join(mimic_folder_path, f"patient_{patient_id}", path)

        class Core():
            root_path = "Core"
            admissions = os.path.join(root_path, "admissions.csv")
            patients = os.path.join(root_path, "patients.csv")
            transfers = os.path.join(root_path, "transfers.csv")

        class CXR_DICOM():
            root_path = "CXR-DICOM"

        class CXR_JPG():
            root_path = "CXR-JPG"
            cxr_chexpert = os.path.join(root_path, "cxr_chexpert.csv")
            cxr_meta = os.path.join(root_path, "cxr_meta.csv")
            cxr_negbio = os.path.join(root_path, "cxr_negbio.csv")
            cxr_split = os.path.join(root_path, "cxr_split.csv")

        class ED():
            root_path = "ED"
            diagnosis = os.path.join(root_path, "diagnosis.csv")
            edstays = os.path.join(root_path, "edstays.csv")
            medrecon = os.path.join(root_path, "medrecon.csv")
            pyxis = os.path.join(root_path, "pyxis.csv")
            triage = os.path.join(root_path, "triage.csv")

        class REFLACX():

            root_path = "REFLACX"
            metadata = os.path.join(root_path, "metadata.csv")

            class REFLACXStudy(Enum):
                anomaly_location_ellipses = "anomaly_location_ellipses.csv"
                chest_bounding_box = "chest_bounding_box.csv"
                fixations = "fixations.csv"
                timestamps_transcription = "timestamps_transcription.csv"
                transcription = "transcription.csv"

                def get_reflacx_path(mimic_folder_path, patient_id, reflacx_id, path):
                    return os.path.join(mimic_folder_path, f"patient_{patient_id}", "REFLACX", reflacx_id, path)

                