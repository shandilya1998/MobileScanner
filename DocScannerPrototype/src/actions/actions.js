import {ADD_SCANNED_PAGE,
        UPDATE_DOCUMENT_CROP} from '../reducers/reducer';

export function onPictureProcessed({originalImage, detectedImage, rectCoord}){
    const payload = {
        'originalImage' : originalImage,
        'detectedDocument' : detectedImage,
        'rectCoords' : rectCoord,
    };
    //console.log(rectCoord);
    return {
        'type' : ADD_SCANNED_PAGE,
        'payload' : payload,
    };
}

export function updatePage(page){
    const payload = {
        'page' : page,
    };
    return {
        'type' : UPDATE_DOCUMENT_CROP,
        'payload' : payload,
    };
}
