import streamlit as st
import hashlib
import json
import os          
from pathlib import Path 
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv
import random, string


load_dotenv()

EMAIL_ADDRESS = os.getenv("EMAIL_ADDRESS")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")

def send_email(to_email, subject, body):
    msg = MIMEMultipart()
    msg['From'] = EMAIL_ADDRESS
    msg['To'] = to_email
    msg['Subject'] = subject

    msg.attach(MIMEText(body, 'plain'))

    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            server.send_message(msg)
        return True
    except Exception as e:
        print(e)
        return False

def hash_password(pwd):
    return hashlib.sha256(pwd.encode()).hexdigest()

def load_users():
    if os.path.exists("users.json"):
        with open("users.json") as f:
            return json.load(f)
    return {}


def save_users(users):
    with open("users.json", "w") as f:
        json.dump(users, f, indent=4)
