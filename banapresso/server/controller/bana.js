import * as banaRepository from '../data/bana.js';

export async function getBanas(req, res) {
    const data = await banaRepository.getAll();
    return res.status(200).json(data);
}