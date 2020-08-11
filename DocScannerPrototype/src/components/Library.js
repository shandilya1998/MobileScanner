import React, {Component} from 'react';
import {View, Text, FlatList, TouchableOpacity} from 'react-native';
import Carousel from 'react-native-snap-carousel';
let RNFS = require('react-native-fs');
import {styles} from '../assets/styles';
import Icon from 'react-native-vector-icons/Ionicons';
const dir = RNFS.ExternalStorageDirectoryPath;


class Library extends Component{
    constructor(props){
        super(props);
        this.state = {
            'currentDir' : {
                'path' : dir,
                'contents' : [],
                'history' : [],
            }
        }
        this.getFolderContentsData = this.getFolderContentsData.bind(this);
        this.setCurrentDir = this.setCurrentDir.bind(this);
        this.renderItem = this.renderItem.bind(this);
        this.listHeaderComponent = this.listHeaderComponent.bind(this);
        this.onPressBack = this.onPressBack.bind(this);
        this.renderPreviousButton = this.renderPreviousButton.bind(this);
    }

    componentDidMount(){
        this.setCurrentDir(this.state.currentDir.path);
    }

    async setCurrentDir(Dir){
        const contents = await RNFS.readDir(dir);
        this.setState(
            ({currentDir})=>{
                return {
                    'currentDir' : {
                        'path' : Dir, 
                        'contents' : contents,
                        'history' : currentDir.history,
                    }
                }
            }
        );
    }
    
    getFolderContentsData(allContents){
        const contents = new Array();
        let i = 0;
        let count = 0;
        for(; i<allContents.length; i++ ){
            if(allContents[i].isFile()){
                const type = allContents[i].name.slice(allContents[i].name.lastIndexOf('.')+1);
                if(type == 'pdf'){
                    const key = count.toString();
                    contents.push({
                        'name' : allContents[i].name,
                        'path' : allContents[i].path,
                        'key' : key,
                        'type' : 'pdf',
                    });
                    count++;
                }
            }
            else if(allContents[i].isDirectory()){
                const key = count.toString();
                contents.push({
                    'name' : allContents[i].name,
                    'path' : allContents[i].path,
                    'key' : key,
                    'type' : 'directory',
                });
                count++;
            } 
        }
        return contents;
    }

    async traverseToDir(Dir){
        const contents = await RNFS.readDir(Dir);
        this.setState(
            ({currentDir})=>{
                currentDir.history.push(currentDir.path);
                return {
                    'currentDir' : { 
                        'path' : Dir, 
                        'contents' : contents,
                        'history' : currentDir.history,
                    }   
                }   
            }   
        );    
    }
 
    onPressBack(){
        const {currentDir} = this.state;
        const Dir = currentDir.history[currentDir.history.length-1];
        currentDir.history.pop();
        RNFS.readDir(Dir).then(
            (contents) => {
                this.setState({
                    'currentDir' : {
                        'path' : Dir,
                        'contents' : contents,
                        'history' : currentDir.history
                    },
                });
            }
        );
    }

    onPressItem(item){
        if(item.type=='directory'){
            this.traverseToDir(item.path);
        }
        else if(item.type=='pdf'){
            //console.log('pdf');
            this.props.onPressPDFFile(item);
        }
    }

    renderItem({item, index, separators}){
        let icon = 'md-document';
        if(item.type == 'pdf'){
            icon = 'md-document';
        }
        else if(item.type == 'directory'){
            icon = 'md-folder';
        }
        return(
            <View
                style = {{
                    alignItems : 'center',
                    justifyContents : 'center',
                    flex : 1,
                    margin : 10,
                    padding : 5
                }}>
                <View
                    style = {[
                        styles.buttonGroup,
                    ]}> 
                    <TouchableOpacity
                        onPress = {()=>this.onPressItem(item)}
                        style = {[
                            styles.button,
                            styles.libraryItem,
                        ]}>
                        <Icon
                            name = {icon}
                            size = {40}
                            color = {'white'}
                            style={[
                                styles.buttonIcon,
                                styles.libraryItemIcon,
                            ]} />
                            <Text style={styles.buttonText}>{item.name}</Text>
                    </TouchableOpacity>
                </View>
            </View>
        );
    }

    listHeaderComponent(){
        return(
            <View 
                style = {styles.libraryHeaderTextContainer}>
                <View
                    style = {styles.libraryHeaderTextContainer}>
                    <Text style = {styles.libraryHeaderTextStyle}>All Files</Text>
                </View>
                <View>
                    {this.renderPreviousButton()}
                </View>
            </View>
        );
    }

    renderPreviousButton(){
        if(this.state.currentDir.history.length<=0){
            return(
                <View
                    style = {
                        {
                            width : '100%',
                            height : 70,
                            flexDirection : 'row',
                            justifyContent : 'space-between',
                            alignItems : 'center',
                            paddingHorizontal : 10,
                        }
                    }>
                    <View
                        style={[
                            styles.buttonGroup,
                            { marginLeft : 8 }
                        ]}/>
                </View>
            );
        }
        else{
            return(
                <View
                    style = {{
                        alignItems : 'center',
                        justifyContents : 'center',
                        flex : 1,
                        margin : 10,
                        padding : 5
                    }}>
                    <View
                        style = {[
                            styles.buttonGroup,
                        ]}>
                        <TouchableOpacity
                            onPress = {()=>this.onPressBack()}
                            style = {[
                                styles.button,
                                {   
                                    height : 35, 
                                    width : 32.5,
                                }
                            ]}>
                            <Icon
                                name = {'md-arrow-round-back'}
                                size = {50}
                                color = {'white'}
                                style={styles.buttonIcon} />
                        </TouchableOpacity>
                    </View>
                </View> 
            );
        } 
    }

    render(){
        //console.log(this.state);
        return(
            <View
                style = {[
                    styles.container,
                    {
                        backgroundColor : 'white',
                        //width : '100%',
                    }
                ]}>
                <View
                    style = {{
                        flex : 1,
                        justifyContent : 'center',
                        alignItems : 'center',
                        width : this.props.width,
                    }}>
                    <FlatList
                        style = {
                            {
                                width : this.props.width,
                            }
                        }
                        data = {this.getFolderContentsData(this.state.currentDir.contents)}
                        ListHeaderComponent = {this.listHeaderComponent()}
                        numColumns = {this.props.numColumns}
                        renderItem = {this.renderItem}/>
                </View>
            </View>
        );
    }
}

export default Library;
