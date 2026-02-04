import streamlit as st
import os
import io
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload

# Config
FOLDER_NAME = "Antigravity_Models"
SCOPES = ['https://www.googleapis.com/auth/drive']
KEY_FILE = "google_key.json"

class DriveManager:
    def __init__(self):
        self.service = self._get_service()
        
    def _get_service(self):
        creds = None
        try:
            # 1. Try Secrets
            if "gcp_service_account" in st.secrets:
                creds = service_account.Credentials.from_service_account_info(
                    st.secrets["gcp_service_account"], scopes=SCOPES)
            
            # 2. Try Local File
            elif os.path.exists(KEY_FILE):
                creds = service_account.Credentials.from_service_account_file(
                    KEY_FILE, scopes=SCOPES)
            
            if creds:
                return build('drive', 'v3', credentials=creds)
        except Exception as e:
            print(f"Auth Error: {e}")
            return None
        return None

    def _get_or_create_folder_id(self):
        if not self.service: return None
        
        try:
            query = f"name='{FOLDER_NAME}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
            results = self.service.files().list(q=query, fields='files(id, name)').execute()
            items = results.get('files', [])
            
            if not items:
                # Create
                metadata = {
                    'name': FOLDER_NAME,
                    'mimeType': 'application/vnd.google-apps.folder'
                }
                file = self.service.files().create(body=metadata, fields='id').execute()
                print(f"Created folder {FOLDER_NAME}")
                return file.get('id')
            else:
                return items[0]['id']
        except Exception as e:
            print(f"Folder Error: {e}")
            return None

    def upload_file(self, local_path):
        """Upload a file to the Antigravity_Models folder."""
        if not self.service: return False, "尚未設定 Google Credentials"
        
        try:
            folder_id = self._get_or_create_folder_id()
            if not folder_id: return False, "無法建立遠端資料夾"
            
            name = os.path.basename(local_path)
            
            # Check if exists (optional: overwrite or version?)
            # Just upload as new version or separate file. Drive allows duplicate names.
            # To be clean, let's just upload. Timestamp in name handles uniqueness.
            
            metadata = {'name': name, 'parents': [folder_id]}
            media = MediaFileUpload(local_path, resumable=True)
            
            file = self.service.files().create(body=metadata, media_body=media, fields='id').execute()
            return True, f"上傳成功 (ID: {file.get('id')})"
        except Exception as e:
            return False, str(e)

    def list_remote_models(self):
        """List models in the folder."""
        if not self.service: return []
        try:
            folder_id = self._get_or_create_folder_id()
            query = f"'{folder_id}' in parents and trashed=false"
            results = self.service.files().list(q=query, orderBy='createdTime desc', fields='files(id, name, createdTime)').execute()
            return results.get('files', [])
        except Exception:
            return []

    def download_file(self, file_id, local_dir):
        """Download a specific file by ID."""
        if not self.service: return False, "No Creds"
        try:
            file_meta = self.service.files().get(fileId=file_id, fields='name').execute()
            name = file_meta.get('name')
            
            request = self.service.files().get_media(fileId=file_id)
            fh = io.FileIO(os.path.join(local_dir, name), 'wb')
            downloader = MediaIoBaseDownload(fh, request)
            
            done = False
            while done is False:
                status, done = downloader.next_chunk()
            
            return True, name
        except Exception as e:
            return False, str(e)
            
    def download_latest_model(self, local_dir):
        """Download the most recent model."""
        files = self.list_remote_models()
        if not files: return False, "雲端無備份模型"
        
        latest = files[0]
        return self.download_file(latest['id'], local_dir)

# Global Instance
drive = DriveManager()
