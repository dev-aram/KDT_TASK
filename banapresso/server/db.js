import MongoDb from 'mongodb'
import { config } from './config.js'

let db;

export async function connectDB() {
    return await MongoDb.MongoClient.connect(config.db.host, {dbName:'kdt'})
    .then((client) => (db = client.db()))
}

export function getBana() {
    return db.collection('bana');
}
