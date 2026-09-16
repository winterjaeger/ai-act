<!-- This is the markdown template for the final project of the Building AI course, 
created by Reaktor Innovations and University of Helsinki. 
Copy the template, paste it to your GitHub README and edit! -->

# Project Title

Final project for the Building AI course

## Summary

ai-act is a Progressive Web App (PWA) that applies Natural Language Processing to facilitate cognitive defusion. By identifying rigid language patterns in daily thought logs, the AI guides users through Acceptance and Commitment Therapy (ACT) exercises to cultivate psychological flexibility.


## Background

Cognitive fusion—the tendency to take thoughts literally and become entangled in them—is a primary driver of psychological distress in Contextual Behavioral Science. While ACT provides robust tools for defusion, applying them in the moment without a practitioner is difficult. My motivation is to engineer a secure, intelligent tool that helps individuals observe their thoughts rather than being dictated by them, fostering internal agency.

The difficulty of applying clinical defusion metaphors independently in daily life.

The lack of privacy-first, accessible digital tools for third-wave CBT practices.

The need for immediate, contextual scaffolding when navigating rigid internal narratives.


## How is it used?

The solution is designed as a PWA for daily use on mobile devices or desktops. Users log their current difficult thoughts or narratives into the interface when they feel stuck. The application analyzes the input for fusion markers and presents contextual defusion exercises. Because these inputs are highly sensitive, the architecture requires client-side zero-knowledge AES-256-GCM encryption to ensure that the server cannot read the raw inputs, maintaining absolute privacy for the user.

import numpy as np

def detect_fusion(user_input, vocabulary, idf_weights, model_coeffs):
    # Process input text into a TF-IDF vector
    words = user_input.lower().split()
    vector = []
    
    for word in vocabulary:
        tf = words.count(word) / len(words) if len(words) > 0 else 0
        idf = idf_weights.get(word, 0)
        vector.append(tf * idf)
        
    # Forward pass through a trained classifier
    z = np.dot(vector, model_coeffs)
    probability = 1 / (1 + np.exp(-z))
    
    if probability > 0.7:
        print("High fusion detected: initiating defusion protocol.")
    else:
        print("Input suggests mindful observation.")


## Data sources and AI methods
The AI component relies on Natural Language Processing (NLP) to classify the degree of cognitive fusion in a text block. The initial model utilizes TF-IDF text analysis combined with a logistic regression classifier, similar to text classification methods used to analyze document similarities.ComponentDescriptionInput DataUser-generated thought logs (encrypted locally).Training DataOpen-source clinical datasets featuring absolute language (e.g., "must", "always", "I am") paired with defused language.AI MethodNLP classification to map input vectors against known fusion archetypes.

## Challenges

This project does not solve or diagnose mental health disorders, nor is it a replacement for professional clinical intervention. The primary ethical consideration is user safety. An AI model might misclassify a statement of severe distress. To mitigate this, the application must maintain a strict boundary as an educational tool rather than a medical device, and the AI defaults to providing neutral, exploratory defusion metaphors rather than prescriptive advice.

## What next?

The project could evolve by integrating local, small-parameter Large Language Models (LLMs) directly into the architecture via WebAssembly. Instead of categorizing text and serving static modules, a local LLM could dynamically generate custom ACT metaphors based on the exact relational frames the user provides.

## Acknowledgments
The foundational principles of cognitive defusion drawn from Acceptance and Commitment Therapy (ACT).
The University of Helsinki and Reaktor Innovations for the Building AI course methodology.
The Association of Contextual Behavioral Science (ACBS)
