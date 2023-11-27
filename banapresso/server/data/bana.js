import MongoDb from 'mongodb';
import { getBana } from '../db.js';

export async function getAll() {
    return getBana()
        .find()
        .toArray()
}