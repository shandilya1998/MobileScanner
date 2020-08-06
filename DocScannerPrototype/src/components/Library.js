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
            }
        }
        this.getFolderContentsData = this.getFolderContentsData.bind(this);
        this.setCurrentDir = this.setCurrentDir.bind(this);
        this.renderItem = this.renderItem.bind(this);
    }

    componentDidMount(){
        this.setCurrentDir(this.state.currentDir.path);
    }

    async setCurrentDir(Dir){
        const contents = await RNFS.readDir(dir);
        this.setState(
            ({currentDir})=>({
                'currentDir' : {
                    ...currentDir,
                    'path' : Dir, 
                    'contents' : contents,
                  }
            })
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
                }}>
                <View
                    style = {[
                        styles.buttonGroup,
                        {backgroundColor : this.state.Image?'#008000' : '#00000080'}
                    ]}> 
                    <TouchableOpacity
                        onPress = {()=>this.onPressImage()}
                        style = {styles.button}>
                        <Icon
                            name = {icon}
                            size = {40}
                            color = {'white'}
                            style={styles.buttonIcon} />
                            <Text style={styles.buttonText}>{item.name}</Text>
                    </TouchableOpacity>
                </View>
            </View>
        );
    }

    render(){
        console.log(this.state);
        return(
            <View
                style = {[
                    styles.container,
                    {
                        backgroundColor : 'white',
                    }
                ]}>
                <View
                    style = {{
                        flex : 1,
                        justifyContent : 'center',
                        alignItems : 'center',
                    }}>
                    <FlatList
                        data = {this.getFolderContentsData(this.state.currentDir.contents)}
                        renderItem = {this.renderItem}/>
                </View>
            </View>
        );
    }
}

export default Library;
