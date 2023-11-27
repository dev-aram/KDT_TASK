import express from 'express';
import morgan from 'morgan';
import cors from 'cors';
import banaRouter from './router/bana.js';
import { config } from './config.js';
import { connectDB } from './db.js';

const app = express();
app.use(express.json());
app.use(morgan('dev'));
app.use(cors());

// 라우터
app.use('/bana',banaRouter);

app.use((req, res, next) => {
    res.sendStatus(404);
  });

connectDB()
    .then((db) => {
    const server = app.listen(config.host.port);
    })
    .catch(console.error);