from __future__ import annotations
import os
from typing import List, Dict
import openai
import pandas as pd
import random
import streamlit as st 
st.write('#Court Room')

from google.colab import drive
drive.mount('/content/drive')


data_path = '/content/drive/My Drive/data.csv'
cases = pd.read_csv(data_path)
case_column = cases.columns[0]
case_list = cases[case_column].dropna().tolist()
case_random = random.choice(case_list)
case_number = case_list.index(case_random)
case_background = case_random  

print("case-number:", case_number, "\n")
print("==== Opening statements ====\n")

class LawyerAgent:
    def __init__(self,
                 name: str,
                 system_prompt: str,
                 model: str = "llama-3.1-8b-instant"):
        self.name = name
        self.system_prompt = system_prompt.strip()
        self.history: List[Dict[str, str]] = []
        self.model = model
        self.client = openai.OpenAI(
            api_key="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",
            base_url="https://api.groq.com/openai/v1"
        )

    def respond(self, user_msg: str, **gen_kwargs) -> str:
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(self.history)
        messages.append({"role": "user", "content": user_msg})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=512,
            temperature=0.7,
            **gen_kwargs
        )
        answer = response.choices[0].message.content.strip()
        self.history.append({"role": "user", "content": user_msg})
        self.history.append({"role": "assistant", "content": answer})
        return answer


class JudgeAgent:
    def __init__(self,
                 name: str,
                 system_prompt: str,
                 model: str = "llama-3.1-8b-instant"):
        self.name = name
        self.system_prompt = system_prompt.strip()
        self.history: List[Dict[str, str]] = []
        self.model = model
        self.client = openai.OpenAI(
            api_key="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",
            base_url="https://api.groq.com/openai/v1"
        )

    def weigh_arguments(self, prosecution_args: str, defense_args: str) -> str:
        user_msg = (
            f"Based on the arguments presented:\n\n"
            f"Prosecution: {prosecution_args}\n\n"
            f"Defense: {defense_args}\n\n"
            f"Please issue a neutral ruling or summary of the strengths/weaknesses of both sides."
        )
        return self.respond(user_msg)

    def give_verdict(self, prosecution_args: str, defense_args: str) -> str:
        user_msg = (
            f"You are now to deliver a verdict. Based on the following:\n\n"
            f"Prosecution's final argument: {prosecution_args}\n"
            f"Defense's final argument: {defense_args}\n\n"
            f"Give a 1-line verdict: 'Guilty (1)' or 'Not Guilty (0)', and a brief reasoning."
        )
        return self.respond(user_msg)

    def respond(self, user_msg: str, **gen_kwargs) -> str:
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(self.history)
        messages.append({"role": "user", "content": user_msg})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=512,
            temperature=0.3,  
            **gen_kwargs
        )
        answer = response.choices[0].message.content.strip()
        self.history.append({"role": "user", "content": user_msg})
        self.history.append({"role": "assistant", "content": answer})
        return answer


# === System Prompts === #

DEFENSE_SYSTEM = """
You are **Alex Carter**, lead *defense counsel*.
Goals: Protect the constitutional rights of the defendant; raise reasonable doubt.
Style: Persuasive and fact-based.
"""

PROSECUTION_SYSTEM = """
You are **Jordan Blake**, *Assistant District Attorney*.
Goals: Present a logical and ethical case for prosecution.
Style: Formal and confident.
"""

DEFENDANT_SYSTEM = """
You are **John Doe**, the *defendant*.
You are accused of a crime and your goal is to maintain innocence unless guilty.
Be emotional, human, and personal — answer questions and make statements as if in real life.
"""

PLAINTIFF_SYSTEM = """
You are **Morgan Fields**, the *plaintiff* in this case.
You believe you were wronged by the defendant.
State your experience clearly and sincerely; speak from a human perspective.
"""

JUDGE_SYSTEM = """
You are **Hon. Rebecca Hayes**, the *presiding judge*.
Remain neutral. Guide the courtroom. Comment only when necessary.
Clarify law, maintain order, and be impartial. Your tone is measured and formal.
and  deliberatly  deliver a verdict based on the arguments,
evidence, and witness testimonies"""

# === Agents === #
defense = LawyerAgent("Defense", DEFENSE_SYSTEM)
prosecution = LawyerAgent("Prosecution", PROSECUTION_SYSTEM)
defendant = LawyerAgent("Defendant", DEFENDANT_SYSTEM)
plaintiff = LawyerAgent("Plaintiff", PLAINTIFF_SYSTEM)
judge = JudgeAgent("Judge", JUDGE_SYSTEM)


# === Opening Statements === #

p_open = prosecution.respond(
    f"Opening statement to the Court. Background: {case_background}"
)
print("PROSECUTOR:", p_open, "\n")

d_open = defense.respond(
    "Opening statement to the Court responding to the prosecution."
)
print("DEFENSE   :", d_open, "\n")

# === Plaintiff Testimony === #

plaintiff_stmt = plaintiff.respond(
    f"Please give your account of the incident based on this case: {case_background}"
)
print("PLAINTIFF :", plaintiff_stmt, "\n")

# === Defendant Response === #

defendant_stmt = defendant.respond(
    f"The plaintiff has given their testimony. Respond to the accusations in this case: {case_background}"
)
print("DEFENDANT :", defendant_stmt, "\n")

# === Prosecutor Rebuttal === #

p_rebut = prosecution.respond("Brief rebuttal to the defense's opening.")
print("PROSECUTOR:", p_rebut, "\n")

# === Final Judgement === #

j_final = judge.respond(
    "Provide a neutral opening comment after hearing both opening statements."
)
print("JUDGE     :", j_final, "\n")