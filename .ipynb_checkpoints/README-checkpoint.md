# Performing clustering on stroke patients based on NIHSS data.

This is the official repository for the research paper 

    Behavioral Clusters and Lesion Distributions in Ischemic Stroke, based on NIHSS Similarity

Submitted to Journal of Healthcare Informatics Research.

**Purpose:**
Stroke is a leading cause of death and disability.
The subsequent dysfunctions relate to brain lesion location.
The complex relationship between behavioral symptoms and lesion location is essential for care and rehabilitation, and useful to understand the healthy brain.
Such complexity eludes linear methods. 
Differently from previous works, this study clusters patients by recurrent symptoms profiles, then retrieves their neural correlates.

**Methods:**
Prevalence, severity and associations of symptoms affecting patients are condensed in a similarity measure between their profiles.
The computation specifically adheres to the ordinal nature of NIHSS data.
A network model links patients with strength proportional to the behavioral similarity.
The network nodes are then classified through a novel spectral clustering variation.
Lesions from each cluster's patients are used in a voxel-wise analysis to statically validate the findings.

**Results:** 
The behavioral clusters express symptoms co-occurring in accordance with the literature on behavioral deficits covariance.
Moreover, they show coherent groups of brain lesions in the associated CT and MRI scans.
The anatomical position of the significant voxels found are aligned with the literature.
Even when lesion density maps overlap, the significant voxels are well separated.

**Conclusion:**
The presented workflow offers concrete, clinically useful correlations between patients profiles, individual patients-to-group associations, in-group co-occurrences of symptoms, and brain lesion correlates. 
The underlying mathematical assumptions, and the unsupervised machine learning techniques  offer statistically robust results and are being further applied to other multimodal biomedical data.


## Authors and Citation
If you find codes and results useful for your research,
please concider citing our work. It would help us to continue our research.


Contributors:

- M.Sc. Louis Fabrice Tshimanga
- M.Sc. Andrea Zanola
- Dr. Silvia Facchini
- Dr. Antonio Luigi Bisogno
- PhD. Lorenzo Pini
- Prof. Manfredo Atzori
- Prof. Maurizio Corbetta

## License

The code is released under the MIT License
