# Course Summary
**D602: Deployment** 

D602 focuses on the practical aspects of deploying data analytics solutions, including analyzing business cases for MLOps, designing and implementing data production pipelines, and deploying APIs. It requires learners to write code, manage it in version control, and present their technical solutions effectively.
# Course Objectives 
WGU outlines the following competencies as a part of this class:
- **Analyzes a Business Case:** The learner analyzes a business case to determine the requirements necessary for deployment.
- **Implements a Data Product Pipeline:** The learner implements a data product pipeline to address organizational needs.
Course Materials This course uses Python or R for coding. GitLab is used for source control. Tools like MLFlow and DVC are specifically mentioned for data pipelines. Dockerfiles are used for API deployment. Panopto video recordings are required for demonstrating the live API.
# Practical Assessment(s) Overview & Files
- **Task 1:** Business Case Analysis
    - **Description:** As a data scientist for a supermarket chain, Kronkers, the task is to analyze the business case for implementing MLOps. This involves describing the objectives of an MLOps deployment architecture, identifying constraints to implementing an MLOps solution, and identifying all functional and non-functional requirements for the proposed MLOps solution.
    - **Output Files:** A business case summary (document file) detailing objectives, constraints, and requirements.
- **Task 2:** Data Production Pipeline
    - **Description:** As an analyst at an airline, the task is to finish a previous analyst's work on predicting flight delays using a polynomial regression model and deploy it to other business units. This involves creating a subgroup and project in GitLab and committing/pushing changes. Learners write a script (Python or R) to import and format downloaded airport data, demonstrating work progression with at least two code versions, and running a DVC command to create and submit a metafile to GitLab. Another script (Python or R) is written to filter data for a chosen airport and implement at least two other data cleaning steps, also with at least two code versions. An MLFlow experiment is implemented to capture features from the poly_regressor file, with at least two code versions. Finally, an MLProject file (using YAML) is written to link these three scripts. A written explanation of the code and MLProject pipeline, including challenges and solutions, and a screenshot of the successful MLProject pipeline run, is required.
    - **Output Files:** GitLab repository containing the import, filter, poly_regressor, and MLProject scripts (with versions), the original CSV data file, and the DVC metafile. A document file with the written explanation and screenshot of the pipeline running successfully.
- **Task 3:** API Deployment
    - **Description:** This task focuses on API deployment. It requires creating a GitLab repository. An API code must be provided where:
        - **The endpoint "/" returns a JSON message indicating the API is functional.**
        - **The endpoint "/predict/delays" provides arrival airport, local departure time, local arrival time, and a JSON response for the average departure delay in minutes.**
    - **Unit tests for the API code must be written (three unit tests in total), including both correctly and incorrectly formatted requests, with at least two versions of the code provided. A Dockerfile is required to package the API code and run a web server to allow HTTP requests, with at least two versions. An explanation of how the code was written, including challenges encountered and addressed, is necessary. Finally, a video demonstration showing one ill-formatted and one well-formatted request, with appropriate API responses, is required.**
    - **Output Files: GitLab repository (containing code versions, unit tests, Dockerfile), a document file with explanations, and a video demonstration of the live API.
