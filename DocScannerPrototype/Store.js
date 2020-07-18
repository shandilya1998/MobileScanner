import {createStore, applyMiddleware} from 'redux';
import thunk from 'redux-thunk';
import reducer from './src/reducers/reducer';

export const store = createStore(reducer, applyMiddleware(thunk));

export const getState = () => store.getState();
