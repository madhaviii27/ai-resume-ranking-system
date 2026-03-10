# AI Resume Ranking System

This project is an AI-powered resume ranking tool that evaluates resumes against a job description using Natural Language Processing.

## Features
- Upload multiple resumes (PDF)
- Input job description
- Rank candidates by similarity score
- Top candidate recommendation
- Interactive dashboard

## Technologies
- Python
- Flask
- Scikit-learn
- TF-IDF Vectorization
- Cosine Similarity
- PyPDF2

## How it works
1. Extract text from uploaded resumes
2. Convert text to TF-IDF vectors
3. Compute similarity between job description and resumes
4. Rank candidates by percentage match
