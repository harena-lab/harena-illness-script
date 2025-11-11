# Arquivos

* annotations_medical_specialist_pre_processed.csv - All the corpus from "teste-progresso" experiment -- containing only the medical specialist annotations (except texts where no annotations were performed)\
**doc_id** - Text ID (Consistent with the initial teste-progresso files);\
**text** - texto completo;\
**labels** - Tokenized text, with annotations and labeling (categories) utilizing BIO format, with word positioning (Tokens that were not annotated are labeled as NONE);\
**minimized_labels** - Tokenized text, with simplified annotations and categories -- contains only token and respective category, lacks token position (Tokens that were not annotated are labeled as NONE);\
**cluster_labels** - Contains only annotated tokens. Simplified format (Token and respective categories).<br />

* test_data_info_no_short.csv - The testing corpus utilized in the testing step for the BioBERTpt approach -- containing only the medical specialist annotations (except texts that don't contain any annotation)\
**doc_id** - Text ID (Consistent with the initial teste-progresso files);\
**text** - Complete text;\
**labels** - Tokenized text, with annotations and labeling (categories) utilizing BIO format, with word positioning (Tokens that were not annotated are labeled as NONE);\
**minimized_labels** - Tokenized text, with simplified annotations and categories -- contains only token and respective category, lacks token position (Tokens that were not annotated are labeled as NONE);\
**cluster_labels** - Contains only annotated tokens. Simplified format (Token and respective categories).<br />

* data_test.csv - Contain informations (text and annotations) from the test corpus utilized in the BioBERTpt testing. (Used this to retrieve the original data to compare with the Llama approach)\
