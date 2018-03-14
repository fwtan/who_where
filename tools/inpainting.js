
// Adobe script for (batch) content aware fill

color_prefix  = TO_BE_SET;
mask_prefix   = TO_BE_SET;
output_prefix = TO_BE_SET;

jpgSaveOptions = new JPEGSaveOptions()
jpgSaveOptions.embedColorProfile = true
jpgSaveOptions.formatOptions = FormatOptions.STANDARDBASELINE
jpgSaveOptions.matte = MatteType.NONE
jpgSaveOptions.quality = 10

color_dir   = new Folder(color_prefix);
color_files = color_dir.getFiles("*.jpg");

for (var i = 0; i < color_files.length; i++) {
  mask_file = new File(mask_prefix+color_files[i].name);
  if (!mask_file.exists) { continue; }

  var maskRef  = app.open(mask_file);
  var colorRef = app.open(color_files[i]);

  // Copy the mask
  app.activeDocument = maskRef
  select_all()
  copy()

  // Paste the mask as the alpha channel
  app.activeDocument = colorRef
  create_alpha_channel()
  paste_to_alpha_channel()

  // Selection from alpha channel
  load_alpha_as_selection()
  show_all_channels()
  set_bg()

  // Content aware fill
  fill(colorRef)
  save_jpeg(colorRef, output_prefix + color_files[i].name);

  // Remember to close the files to release memory
  // colorRef.close(SaveOptions.DONOTSAVECHANGES);
  // maskRef.close(SaveOptions.DONOTSAVECHANGES);
  close_file(colorRef)
  close_file(maskRef)
}

function save_jpeg(doc_ref, path) {
  app.activeDocument = doc_ref
  jpgFile = new File(path)
  app.activeDocument.saveAs(jpgFile, jpgSaveOptions, true,
                 Extension.LOWERCASE)
  jpgFile.close(SaveOptions.DONOTSAVECHANGES)
}

function fill(doc_ref) {
  app.activeDocument = doc_ref
  // =======================================================
  var idFl = charIDToTypeID( "Fl  " );
      var desc_fill = new ActionDescriptor();
      var idUsng = charIDToTypeID( "Usng" );
      var idFlCn = charIDToTypeID( "FlCn" );
      var idcontentAware = stringIDToTypeID( "contentAware" );
      desc_fill.putEnumerated( idUsng, idFlCn, idcontentAware );
      var idcontentAwareColorAdaptationFill = stringIDToTypeID( "contentAwareColorAdaptationFill" );
      desc_fill.putBoolean( idcontentAwareColorAdaptationFill, true );
      var idOpct = charIDToTypeID( "Opct" );
      var idPrc = charIDToTypeID( "#Prc" );
      desc_fill.putUnitDouble( idOpct, idPrc, 100.000000 );
      var idMd = charIDToTypeID( "Md  " );
      var idBlnM = charIDToTypeID( "BlnM" );
      var idNrml = charIDToTypeID( "Nrml" );
      desc_fill.putEnumerated( idMd, idBlnM, idNrml );
  executeAction( idFl, desc_fill, DialogModes.NO );
}

function select_all() {

  var idsetd = charIDToTypeID( "setd" );
    var desc_sel_all = new ActionDescriptor();
    var idnull = charIDToTypeID( "null" );
        var ref1 = new ActionReference();
        var idChnl = charIDToTypeID( "Chnl" );
        var idfsel = charIDToTypeID( "fsel" );
        ref1.putProperty( idChnl, idfsel );
    desc_sel_all.putReference( idnull, ref1 );
    var idT = charIDToTypeID( "T   " );
    var idOrdn = charIDToTypeID( "Ordn" );
    var idAl = charIDToTypeID( "Al  " );
    desc_sel_all.putEnumerated( idT, idOrdn, idAl );
  executeAction( idsetd, desc_sel_all, DialogModes.NO );
}

function copy() {
  var idcopy = charIDToTypeID( "copy" );
  executeAction( idcopy, undefined, DialogModes.NO );
}

function create_alpha_channel() {
  var idMk = charIDToTypeID( "Mk  " );
    var desc19 = new ActionDescriptor();
    var idNw = charIDToTypeID( "Nw  " );
        var desc20 = new ActionDescriptor();
        var idClrI = charIDToTypeID( "ClrI" );
        var idMskI = charIDToTypeID( "MskI" );
        var idMskA = charIDToTypeID( "MskA" );
        desc20.putEnumerated( idClrI, idMskI, idMskA );
        var idClr = charIDToTypeID( "Clr " );
            var desc21 = new ActionDescriptor();
            var idRd = charIDToTypeID( "Rd  " );
            desc21.putDouble( idRd, 255.000000 );
            var idGrn = charIDToTypeID( "Grn " );
            desc21.putDouble( idGrn, 0.000000 );
            var idBl = charIDToTypeID( "Bl  " );
            desc21.putDouble( idBl, 0.000000 );
        var idRGBC = charIDToTypeID( "RGBC" );
        desc20.putObject( idClr, idRGBC, desc21 );
        var idOpct = charIDToTypeID( "Opct" );
        desc20.putInteger( idOpct, 50 );
    var idChnl = charIDToTypeID( "Chnl" );
    desc19.putObject( idNw, idChnl, desc20 );
  executeAction( idMk, desc19, DialogModes.NO );
}

function paste_to_alpha_channel() {
  var idpast = charIDToTypeID( "past" );
      var desc24 = new ActionDescriptor();
      var idAntA = charIDToTypeID( "AntA" );
      var idAnnt = charIDToTypeID( "Annt" );
      var idAnno = charIDToTypeID( "Anno" );
      desc24.putEnumerated( idAntA, idAnnt, idAnno );
      var idAs = charIDToTypeID( "As  " );
      var idPxel = charIDToTypeID( "Pxel" );
      desc24.putClass( idAs, idPxel );
  executeAction( idpast, desc24, DialogModes.NO );

}

function load_alpha_as_selection() {
  var idsetd = charIDToTypeID( "setd" );
      var desc25 = new ActionDescriptor();
      var idnull = charIDToTypeID( "null" );
          var ref7 = new ActionReference();
          var idChnl = charIDToTypeID( "Chnl" );
          var idfsel = charIDToTypeID( "fsel" );
          ref7.putProperty( idChnl, idfsel );
      desc25.putReference( idnull, ref7 );
      var idT = charIDToTypeID( "T   " );
          var ref8 = new ActionReference();
          var idChnl = charIDToTypeID( "Chnl" );
          var idOrdn = charIDToTypeID( "Ordn" );
          var idTrgt = charIDToTypeID( "Trgt" );
          ref8.putEnumerated( idChnl, idOrdn, idTrgt );
      desc25.putReference( idT, ref8 );
  executeAction( idsetd, desc25, DialogModes.NO );
}

function show_all_channels() {
  var idShw = charIDToTypeID( "Shw " );
      var desc26 = new ActionDescriptor();
      var idnull = charIDToTypeID( "null" );
          var list1 = new ActionList();
              var ref9 = new ActionReference();
              var idChnl = charIDToTypeID( "Chnl" );
              var idChnl = charIDToTypeID( "Chnl" );
              var idRd = charIDToTypeID( "Rd  " );
              ref9.putEnumerated( idChnl, idChnl, idRd );
          list1.putReference( ref9 );
              var ref10 = new ActionReference();
              var idChnl = charIDToTypeID( "Chnl" );
              var idChnl = charIDToTypeID( "Chnl" );
              var idGrn = charIDToTypeID( "Grn " );
              ref10.putEnumerated( idChnl, idChnl, idGrn );
          list1.putReference( ref10 );
              var ref11 = new ActionReference();
              var idChnl = charIDToTypeID( "Chnl" );
              var idChnl = charIDToTypeID( "Chnl" );
              var idBl = charIDToTypeID( "Bl  " );
              ref11.putEnumerated( idChnl, idChnl, idBl );
          list1.putReference( ref11 );
      desc26.putList( idnull, list1 );
  executeAction( idShw, desc26, DialogModes.NO );

}

function set_bg() {
  var idsetd = charIDToTypeID( "setd" );
      var desc27 = new ActionDescriptor();
      var idnull = charIDToTypeID( "null" );
          var ref12 = new ActionReference();
          var idLyr = charIDToTypeID( "Lyr " );
          var idBckg = charIDToTypeID( "Bckg" );
          ref12.putProperty( idLyr, idBckg );
      desc27.putReference( idnull, ref12 );
      var idT = charIDToTypeID( "T   " );
          var desc28 = new ActionDescriptor();
          var idOpct = charIDToTypeID( "Opct" );
          var idPrc = charIDToTypeID( "#Prc" );
          desc28.putUnitDouble( idOpct, idPrc, 100.000000 );
          var idMd = charIDToTypeID( "Md  " );
          var idBlnM = charIDToTypeID( "BlnM" );
          var idNrml = charIDToTypeID( "Nrml" );
          desc28.putEnumerated( idMd, idBlnM, idNrml );
      var idLyr = charIDToTypeID( "Lyr " );
      desc27.putObject( idT, idLyr, desc28 );
      var idLyrI = charIDToTypeID( "LyrI" );
      desc27.putInteger( idLyrI, 3 );
  executeAction( idsetd, desc27, DialogModes.NO );

}

function close_file(doc_ref) {
  app.activeDocument = doc_ref
  var idCls = charIDToTypeID( "Cls " );
      var desc_close = new ActionDescriptor();
      var idSvng = charIDToTypeID( "Svng" );
      var idYsN = charIDToTypeID( "YsN " );
      var idN = charIDToTypeID( "N   " );
      desc_close.putEnumerated( idSvng, idYsN, idN );
      var idDocI = charIDToTypeID( "DocI" );
      desc_close.putInteger( idDocI, 195 );
      var idforceNotify = stringIDToTypeID( "forceNotify" );
      desc_close.putBoolean( idforceNotify, true );
  executeAction( idCls, desc_close, DialogModes.NO );

}
