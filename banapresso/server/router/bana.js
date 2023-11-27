import express from "express";
import * as banaController from '../controller/bana.js';

const router = express.Router()

router.get("/", banaController.getBanas)

export default router;