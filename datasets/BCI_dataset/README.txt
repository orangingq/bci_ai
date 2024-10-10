## BCI_dataset

### Directory Structure

|--BCI_dataset  
	|--HE
	       |--train
	       |--test
	|--IHC  
	       |--train
	       |--test
	README.md

### Statistics

Our BCI dataset contains 9746 images (4873 pairs), 3896 pairs for train and 977 for test.  
The naming convention for images is 'number_train/test_HER2level.png'. Note that the HER2 level represents which category of WSI the image came from, not the image itself.
The same pair of HE and IHC images have the same number.  

### License

This BCI Dataset is made freely available to academic and non-academic entities for non-commercial purposes such as academic research, teaching, scientific publications, or personal experimentation. Permission is granted to use the data given that you agree to our license terms bellow:

1. That you include a reference to the BCI Dataset in any work that makes use of the dataset. For research papers, cite our preferred publication as listed on our website; for other media cite our preferred publication as listed on our website or link to the BCI website.
2. That you may not use the dataset or any derivative work for commercial purposes as, for example, licensing or selling the data, or using the data with a purpose to procure a commercial gain.
3. That all rights not expressly granted to you are reserved by us.

### Privacy

All data are desensitized, and the private information has been removed. If you have any privacy concerns, please contact us by sending an e-mail to shengjie.Liu@bupt.edu.cn, czhu@bupt.edu.cn, or bupt.ai.cz@gmail.com.

### Citation

If you use this data for your research, please cite our paper: BCI: Breast Cancer Immunohistochemical Image Generation through Pyramid Pix2pix

@article{liu2022bci,
  title={BCI: Breast Cancer Immunohistochemical Image Generation through Pyramid Pix2pix},
  author={Liu, Shengjie and Zhu, Chuang and Xu, Feng and Jia, Xinyu and Shi, Zhongyue and Jin, Mulan},
  journal={arXiv preprint arXiv:2204.11425},
  year={2022}
}