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
    print(f"EMAIL SIMULATA:\nA: {to_email}\nOggetto: {subject}\nCorpo:\n{body}")
    return True

#def send_email(to_email, subject, body):
#    msg = MIMEMultipart()
#    msg['From'] = EMAIL_ADDRESS
#    msg['To'] = to_email
#    msg['Subject'] = subject
#    msg.attach(MIMEText(body, 'plain'))

#    try:
#        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
#            server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
#            server.send_message(msg)
#        return True
#    except Exception as e:
#        print("Errore invio email:", e)  # Mostra l’errore reale
#        return False


#def hash_password(pwd):
 #   return hashlib.sha256(pwd.encode()).hexdigest()

#def load_users():
#    if os.path.exists("users.json"):
#        with open("users.json") as f:
#            return json.load(f)
#    return {}


#def save_users(users):
#    with open("users.json", "w") as f:
#        json.dump(users, f, indent=4)


import gspread
from oauth2client.service_account import ServiceAccountCredentials


# Autenticazione
scope = ["https://spreadsheets.google.com/feeds",'https://www.googleapis.com/auth/drive']
creds = ServiceAccountCredentials.from_json_keyfile_name("service_account.json", scope)
client = gspread.authorize(creds)

# Apri sheet
sheet = client.open("FinEdu Users").sheet1

# -----------------------------
# Funzioni utili
# -----------------------------
def hash_password(pwd):
    return hashlib.sha256(pwd.encode()).hexdigest()

def load_users():
    users = {}
    rows = sheet.get_all_records()  # lista di dizionari
    for row in rows:
        users[row['Username']] = {"password": row['Password'], "email": row['Email']}
    return users

def save_user(username, password_hash, email):
    # Aggiunge una nuova riga
    sheet.append_row([username, password_hash, email])

