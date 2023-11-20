# pip install python-multipart

from fastapi import FastAPI, File, UploadFile

app = FastAPI()

# 바이트 단위로 데이터를 전달 받음
# 메모리에 저장됨
@app.post("/files")
async def create_file(file: bytes | None = File(default=None)):
    if not file:
        return {"message": "파일이 존재하지 않음"}
    else:
        return {"file_size":len(file)}


# File()를 따로 생성할 필요가 없음
# 만약 파일의 크기가 메모리 사이즈 이상을 넘어가면 디스크에 저장
# 이미지, 비디오, 큰 바이너리 파일 등과 같은 파일에 적합
# 메타데이터 정보도해당 구문으로 통해 얻을 수 있음
# @app.post("/uploadfile")
# async def create_upload_file(file: UploadFile):
#     if not file:
#         return {"message": "파일이 존재하지 않음"}
#     else:
#         # return {"filename": file.filename}
#         return {"file": file}
    

@app.post("/uploadfile")
async def create_upload_file(file: UploadFile = File(...)):
    if not file:
        return {"message": "파일이 존재하지 않음"}
    else:
        # 파일을 저장할 경로 설정 (예: 현재 작업 디렉토리의 files 폴더에 저장됨)
        upload_folder = "files"
        file_path = f"{upload_folder}/{file.filename}"
        # 파일을 원하는 경로에 저장한다
        with open(file_path, "wb") as f:
            f.write(file.file.read())
        return {"filename": file.filename, "saved_path": file_path}