from pathlib import Path
import llms
import core
import gradio as gr

# add more roles depending on the company dataset.
# for this project, I used dataset job categories :)
job_roles = [
    "ACCOUNTANT", "ADVOCATE", "AGRICULTURE", "APPAREL", "ARCHITECTURE", "ARTS", "AUTOMOBILE", "AVIATION", "BANKING", "BLOCKCHAIN",
    "Business Process Outsourcing", "BUILDING AND CONSTRUCTION", "BUSINESS ANALYST", "BUSINESS DEVELOPMENT", "CHEF", "CIVIL ENGINEER",
    "CONSTRUCTION", "CONSULTANT", "DATA SCIENCE", "DATABASE", "DESIGNER", "DESIGNING", "DEVOPS", "DIGITAL MEDIA", "DOTNET DEVELOPER",
    "EDUCATION", "ELECTRICAL ENGINEERING", "ENGINEERING", "ETL DEVELOPER", "FINANCE", "FITNESS", "FOOD AND BEVERAGES",
    "HEALTH AND FITNESS", "HEALTHCARE", "HUMAN RESOURCES", "INFORMATION TECHNOLOGY", "JAVA DEVELOPER", "MANAGEMENT",
    "MECHANICAL ENGINEER", "NETWORK SECURITY ENGINEER", "OPERATIONS MANAGER", "Project Management Office", "PUBLIC RELATIONS",
    "PYTHON DEVELOPER", "REACT DEVELOPER", "SALES", "SAP DEVELOPER", "SQL DEVELOPER", "TEACHER", "TESTING", "WEB DESIGNING"
]
model_name = "meta-llama/Llama-3.2-1B-Instruct"


def analyze_resume(resume_file, job_description_file, job_description_str, role, number_candidate):
    global data_path
    if not job_description_file and not role:
        return "Please upload a file or enter a job skills.", \
            "Please upload a file or enter a job skills.", \
            "Please upload a file or enter a job skills.", \
            "Please upload a file or enter a job skills.", \
            "Please upload a file or enter a job skills."

    # Load text from file Support short documents ONLY
    if job_description_file is not None:
        job_description_file = core.load_file(job_description_file.name)
    else:
        job_description_file = ""

    # Load text from file
    if resume_file is not None:
        resume_file = core.load_file(job_description_file.name)
    else:
        resume_file = ""

    # Call RAG + LLM pipeline
    resume_text_db = core.load_db(data_path)
    # sample = core.rags.get_sample(resume_text_db, 0)
    # requirements = sample["sample"]["Category"]
    requirements = f"{job_description_file}\n{job_description_str}\nRole: {role}"
    candidates = core.get_candidates_given_job(requirements,
                                               resume_text_db,
                                               model_name,
                                               )
    if len(candidates) == 0:
        return "No Candiadetes in our DB were found.", \
            "No Candiadetes in our DB were found.", \
            "No Candiadetes in our DB were found.", \
            "No Candiadetes in our DB were found.", \
            "No Candiadetes in our DB were found."

    score = ""
    rclass = ""
    summary = ""
    errors = ""
    resume = ""
    for idx, (cand_idx, candidate) in enumerate(candidates.items()):
        score += f"{idx+1}. Candidate {cand_idx}:  " + \
            candidate.get("RECOMMENDED_SCORE", "N/A") + " \n\n"
        rclass += f"{idx+1}. Candidate {cand_idx}:  " + \
            candidate.get("RECOMMENDED_CLASS", "N/A") + "\n\n"
        summary += f"{idx+1}. Candidate {cand_idx}:  " + \
            candidate.get("SUMMARY", "No summary available.") + "\n\n"
        errors += f"{idx+1}. Candidate {cand_idx}:  " + \
            candidate.get("ERRORS_OR_INCONSISTENCIES",
                          "No Errors were found.") + "\n\n"
        sample = core.rags.get_sample(resume_text_db, int(cand_idx))["sample"]
        resume += f"{idx+1}. Candidate: {cand_idx} - label: {sample['Category']}:\n" + \
            sample['Resume'] + "\n\n\n\n"
        if idx+1 == int(number_candidate):
            break
    return score, rclass, summary, errors, resume


def main():
    with gr.Blocks() as interface:
        gr.Markdown("# Resume Analyzer")

        with gr.Row():
            resume_input = gr.File(
                label="Upload Resume for evaluation (.txt, .pdf)", file_types=[".txt", ".pdf"])
            job_description_input = gr.File(
                label="Upload Job Role (.txt, .pdf)", file_types=[".txt", ".pdf"])
            job_description_str_input = gr.Textbox(
                label="Target Job Skills", placeholder="e.g., software skills, Excel, Python, leadership",
                lines=7)
            with gr.Column():
                role_input = gr.Dropdown(
                    label="Select Target Job Role", choices=job_roles)
                candidate_input = gr.Dropdown(
                    label="Select Number of Candidates to Return",
                    choices=list(range(1, 21)),
                    value=1,
                    allow_custom_value=True
                )

        with gr.Row():
            submit_btn = gr.Button("Analyze Resume")

        with gr.Column():
            with gr.Row():
                score_output = gr.Textbox(label="Compatibility Score")
                rclass_output = gr.Textbox(label="Recommended Job Positions")
                summary_output = gr.Textbox(label="Summary")
                errors_output = gr.Textbox(label="Errors")
            cv_output = gr.Textbox(label="Candidate Resumes", lines=25)

        submit_btn.click(
            analyze_resume,
            inputs=[resume_input,
                    job_description_input,
                    job_description_str_input,
                    role_input,
                    candidate_input],
            outputs=[score_output,
                     rclass_output,
                     summary_output,
                     errors_output,
                     cv_output]
        )

    interface.launch(share=False)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run Resume Analyzer")
    parser.add_argument("--db_path", type=str,
                        default="embeddings/resume_classifier/embeddingsBase/chroma/train/Resume",
                        help="Path to load the Resume dataset")
    args = parser.parse_args()
    data_path = args.db_path
    main()
