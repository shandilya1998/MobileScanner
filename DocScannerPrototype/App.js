import React, {Component} from 'react';
import MainApp from './src/MainApp';
import {Provider} from 'react-redux';
import {store} from './Store';

class App extends Component{
    constructor(props){
        super(props);
        console.disableYellowBox = true;
    }

    render(){
        return (
            <Provider store = {store}>
                <MainApp />
            </Provider>
        );
    }
}

export default App;
