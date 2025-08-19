# onto_rag

A Retrieval Augmented Generation system for matching extracted entities to ontologies

## TODO

### Model Improvements

- [] Do not call the scorer if there are no good candidates
- [ ] Integrate scorer recommendations into the synonym generator
- [ ] Merge the scorer and synonym generator into a unified model
- [ ] Optimize and minimize prompt templates for better efficiency
- [x] ±100-char context window
- [ ] Add the scorer feedback to the loop for iterative improvement
- [ ] If confidence is still low switch to a stronger model

### Performance & Infrastructure

- [ ] Implement caching mechanism for improved response times
- [ ] Experiment with different embedding models for better accuracy
- [ ] Add pre-flight exact match checking to avoid unnecessary processing

### Testing & Evaluation

- [ ] Complete evaluation runs on the Cafeteria dataset
- [ ] Validate system performance with comprehensive test cases