import dotenv from 'dotenv';
dotenv.config();

function required(key, defualtValue = undefined) {
    // defualtValue에 undefined로 초기화
    const value = process.env[key] || defualtValue; // process.env[key]담기고  없으면 defualtValue(undefined) 담긴다.
    if (value == null) {
      // 값이 null일때
      throw new Error(`Key ${key} is undefind`); // 에러메세지
    }
    return value;
}

export const config = {
    host: {
        port: parseInt(required('HOST_PORT', 8080)),
    },
    db: {
        host: required('DB_HOST'),
    },
};