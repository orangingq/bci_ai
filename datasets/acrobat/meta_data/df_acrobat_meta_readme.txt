ACROBAT DATASET README

This file provides details on the columns in df_acrobat_meta.csv.

The ACROBAT data set consists of 4,212 whole slide images (WSIs) from 1,153 female primary breast cancer patients. The WSIs in the data set are available at 10X magnification and show tissue sections from breast cancer resection specimens stained with hematoxylin and eosin (H&E) or immunohistochemistry (IHC). For each patient, one WSI of H&E stained tissue and at least one one, and up to four, WSIs of corresponding tissue stained with the routine diagnostic stains ER, PGR, HER2 and KI67 are available. The data set was acquired as part of the CHIME study (chimestudy.se) and its primary purpose was to facilitate the ACROBAT WSI registration challenge (acrobat.grand-challenge.org). The histopathology slides originate from routine diagnostic pathology workflows and were digitised for research purposes at Karolinska Institutet (Stockholm, Sweden). The image acquisition process resembles the routine digital pathology image digitisation workflow, using three different Hamamatsu WSI scanners, specifically one NanoZoomer S360 and two NanoZoomer XR. The WSIs in this data set are accompanied by a data table with one row for each WSI, specifying an anonymised patient ID, the stain or IHC antibody type of each WSI, as well as the magnification and microns per pixel at each available resolution level. Automated registration algorithm performance evaluation is possible through the ACROBAT challenge website based on over 37,000 landmark pair annotations from 13 annotators. While the primary purpose of this data set was the development and evaluation of WSI registration methods, this data set has the potential to facilitate further research in the context of computational pathology, for example in the areas of stain-guided learning, virtual staining, unsupervised learning and stain-independent models. 


COLUMN NAME               DEFINITION
anon_id                   Anonymized case id. The same case ids can occur in the training, validation and test set, so combine this with the set column to uniquely identify a patient.
set                       Indicates which data split this file belongs to.
stain                     Indicates the stain for WSIs of H&E stained tissue or IHC antibody for WSIs of IHC stained tissue.
filename                  Indicates the filename that the meta data refers to.
vendor                    Vendor of the scanner used to generate the WSI. Always Hamamatsu in this dataset.
model                     Indicates the Hamamatsu scanner model used to generate the respective WSI.
mpp_lvl_X,                Indicates the microns-per-pixel for pyramidal level X, where X is in [0, 8]. Not all WSIs have 9 levels, the value if a level does not exist is NaN.
magnification_lvl_X       Indicates the magnification for pyramidal level X, where X is in [0, 8], starting at 10X. Not all WSIs have 9 levels, the value if a level does not exist is NaN.
