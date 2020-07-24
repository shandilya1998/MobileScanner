export const ADD_SCANNED_PAGE = 'mobscan_add_scanned_page';
export const UPDATE_DOCUMENT_CROP = 'mobscan_updt_scan_doc_crop';
export const FLUSH_SCANNED_DOCUMENT = 'mobscan_flush_scanned_doc';

export const initialState = {
    scannedDocument : [],
};

export default function reducer(state = initialState, action){
    switch(action.type){
        case ADD_SCANNED_PAGE:
            const updatedScannedDocument = [
                ...state.scannedDocument,
                {
                    'originalImage' : action.payload.originalImage,
                    'detectedDocument' : action.payload.detectedDocument,
                    'rectCoords' : action.payload.rectCoords,
                    'pageNum' : state.scannedDocument.length + 1,
                }        
            ];
            //console.log(updatedScannedDocuments.length);
            return {
                ...state,
                'scannedDocument' : updatedScannedDocument,
            };
        case UPDATE_DOCUMENT_CROP:
            return {
                ...state,
                'scannedDocument' : action.payload['doc'],
            };
        case FLUSH_SCANNED_DOCUMENT:
            return {
                ...state,
                'scannedDocument' : [],
            };
        default:
            return {...state};
    }
}
