'''
viewprobe creates visualizations for a certain eval.
'''

import re
import numpy
from imageio import imread, imwrite
from skimage.transform import resize
from PIL import Image
import visualize.expdir as expdir
import visualize.bargraph as bargraph
import settings
import numpy as np
from skimage.color import gray2rgb
from skimage import img_as_ubyte
from matplotlib import pyplot as plt
# unit,category,label,score

replacements = [(re.compile(r[0]), r[1]) for r in [
    (r'-[sc]$', ''),
    (r'_', ' '),
    ]]

def fix(s):
    for pattern, subst in replacements:
        s = re.sub(pattern, subst, s)
    return s

# get code(from category_label dict mapping) from label name for further processing in segementation map
def get_code_by_label_name(ds, category, label_name):
    for label_info in ds.category_label[category]:
        if label_info['name'] == label_name:
            return label_info['code']
    return None


def generate_html_summary(ds, layer, maxfeature=None, features=None, thresholds=None,
        imsize=None, imscale=72, tally_result=None,
        gridwidth=None, gap=3, limit=None, force=False, verbose=False):
    ed = expdir.ExperimentDirectory(settings.OUTPUT_FOLDER)
    print('Generating html summary %s' % ed.filename('html/%s.html' % expdir.fn_safe(layer)))
    # Grab tally stats
    # bestcat_pciou, name_pciou, score_pciou, _, _, _, _ = (tally_stats)
    if verbose:
        print('Sorting units by score.')
    if imsize is None:
        imsize = settings.IMG_SIZE
    top = np.argsort(maxfeature, 0)[:-1 - settings.TOPN:-1, :].transpose()
    ed.ensure_dir('html','image')
    html = [html_prefix]
    rendered_order = []
    barfn = 'image/%s-bargraph.svg' % (
            expdir.fn_safe(layer))
    bargraph.bar_graph_svg(ed, layer,
                           tally_result=tally_result,
                           rendered_order=rendered_order,
                           save=ed.filename('html/' + barfn))
    html.extend([
        '<div class="histogram">',
        '<img class="img-fluid" src="%s" title="Summary of %s %s">' % (
            barfn, ed.basename(), layer),
        '</div>'
        ])
    html.append('<div class="gridheader">')
    html.append('<div class="layerinfo">')
    html.append('%d/%d units covering %d concepts with IoU &ge; %.2f' % (
        len([record for record in rendered_order
            if float(record['score']) >= settings.SCORE_THRESHOLD]),
        len(rendered_order),
        len(set(record['label'] for record in rendered_order
            if float(record['score']) >= settings.SCORE_THRESHOLD)),
        settings.SCORE_THRESHOLD))
    html.append('</div>')
    html.append(html_sortheader)
    html.append('</div>')

    if gridwidth is None:
        gridname = ''
        gridwidth = settings.TOPN
        gridheight = 1 
    else:
        gridname = '-%d' % gridwidth
        gridheight = (settings.TOPN + gridwidth - 1) // gridwidth

    html.append('<div class="unitgrid"') # Leave off > to eat spaces
    if limit is not None:
        rendered_order = rendered_order[:limit]
    for i, record in enumerate(
            sorted(rendered_order, key=lambda record: -float(record['score']))):
        record['score-order'] = i
    for label_order, record in enumerate(rendered_order):
        unit = int(record['unit']) - 1 # zero-based unit indexing
        imfn = 'image/%s%s-%04d.jpg' % (
                expdir.fn_safe(layer), gridname, unit)
        category = record['category']
        label = record['label']
        # get code of label from category mapping
        label_code = get_code_by_label_name(ds, category=category, label_name = label)
        img_path = {}
        if force or not ed.has('html/%s' % imfn):
            if verbose:
                print('Visualizing %s unit %d' % (layer, unit))
            # Generate the top-patch image
            tiled = numpy.full(
                ((imsize + gap) * gridheight * 2 - gap,  # 乘以 2 是因为要显示两排：一排原图，一排mask
                (imsize + gap) * gridwidth - gap, 3), 255, dtype='uint8')
            for x, index in enumerate(top[unit]):
                img_path[x] = index
                row = x // gridwidth
                col = x % gridwidth
                image = imread(ds.filename(index))
                # 这里使用 order=1 来指定双线性插值，mode='reflect' 处理边界外像素
                ft_mask = resize(features[index][unit], image.shape[:2], order=1, mode='reflect')
                ft_mask = ft_mask > thresholds[unit]
                ft_vis = (ft_mask[:, :, numpy.newaxis] * 0.8 + 0.2) * image
                if ft_vis.shape[:2] != (imsize, imsize):
                    ft_vis = resize(ft_vis, (imsize, imsize))
                
                gt_mask = ds.segmentation_data(category, index)
                # preserve label we want from segementaion map and set other to 0
                gt_mask_singel_label = np.where(gt_mask == label_code, 1, 0) # binary mask

                ## due to the gt_mask is represented in segementation code, we can't do resize with interpolation first or after, which results in weird highlight
                ## binlinear interpolation. One example: mask: (112, 112, 1), original:(224, 224, 3)
                # gt_mask_singel_label = resize(gt_mask_singel_label, image.shape[:2], order=1, mode='reflect', preserve_range=True)
                
                # order=0 for boolean image
                gt_mask_singel_label = resize(gt_mask_singel_label, image.shape[:2], order=0, mode='reflect')

                gt_vis = (gt_mask_singel_label[:, :, numpy.newaxis] * 0.8 + 0.2) * image
                if gt_vis.shape[:2] != (imsize, imsize):
                    gt_vis = resize(gt_vis, (imsize, imsize))

                # Place the processed image in the tiled array
                tiled[row * (imsize + gap) * 2:row * (imsize + gap) * 2 + imsize,
                      col * (imsize + gap):col * (imsize + gap) + imsize, :] = ft_vis
              
                # Place the ground truth mask right below the corresponding image
                tiled[(row * 2 + 1) * (imsize + gap):(row * 2 + 1) * (imsize + gap) + imsize,
                      col * (imsize + gap):col * (imsize + gap) + imsize, :] = gt_vis
            # Save the entire tiled image as one composite image
            imwrite(ed.filename('html/' + imfn), tiled)
        # Generate the wrapper HTML
        img_path_str = ', '.join(f'{k}: {v}' for k, v in img_path.items())
        graytext = ' lowscore' if float(record['score']) < settings.SCORE_THRESHOLD else ''
        html.append('><div class="unit%s" data-order="%d %d %d">' %
                (graytext, label_order, record['score-order'], unit + 1))
        html.append('<div class="unitlabel">%s</div>' % fix(record['label']))
        html.append('<div class="info">' +
            '<span class="layername">%s</span> ' % layer +
            '<span class="unitnum">unit %d</span> ' % (unit + 1) +
            '<span class="category">(%s)</span> ' % record['category'] +
            '<span class="iou">IoU %.2f</span>' % float(record['score']) + 
            '<span class="full-index" style="display:none;">%s</span>' % img_path_str +
            '</div>')
        html.append('<span class="img"></span>')
        html.append(
            '<div class="thumbcrop"><img src="%s" height="%d"></div>' %
            (imfn, imscale))
        html.append('</div') # Leave off > to eat spaces
    html.append('></div>')
    html.extend([html_suffix]);
    with open(ed.filename('html/%s.html' % expdir.fn_safe(layer)), 'w') as f:
        f.write('\n'.join(html))

html_prefix = '''
<!doctype html>
<html>
<head>
<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0-alpha.6/css/bootstrap.min.css">
<script src="https://code.jquery.com/jquery-3.2.1.min.js" integrity="sha256-hwg4gsxgFZhOsEEamdOYGBf13FyQuiTwlAQgxVSNgt4=" crossorigin="anonymous"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/tether/1.4.0/js/tether.min.js"></script>
<script src="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0-alpha.6/js/bootstrap.min.js" integrity="sha384-vBWWzlZJ8ea9aCX4pEW3rVHjgjt7zpkNpZk+02D9phzyeVkE+jo0ieGizqPLForn" crossorigin="anonymous"></script>
<style>
.unitviz, .unitviz .modal-header, .unitviz .modal-body, .unitviz .modal-footer {
  font-family: Arial;
  font-size: 15px;
}
.unitgrid {
  text-align: center;
  border-spacing: 5px;
  border-collapse: separate;
}
.unitgrid .info {
  text-align: left;
}
.unitgrid .layername {
  display: none;
}
.unitlabel {
  font-weight: bold;
  font-size: 150%;
  text-align: center;
  line-height: 1;
}
.lowscore .unitlabel {
   color: silver;
}
.thumbcrop {
  overflow: hidden;
  width: 288px;
  height: 72px;
}
.unit {
  display: inline-block;
  background: white;
  padding: 3px;
  margin: 2px;
  box-shadow: 0 5px 12px grey;
}
.iou {
  display: inline-block;
  float: right;
}
.modal .big-modal {
  width:auto;
  max-width:90%;
  max-height:80%;
}
.modal-title {
  display: inline-block;
}
.footer-caption {
  float: left;
  width: 100%;
}
.histogram {
  text-align: center;
  margin-top: 3px;
}
.img::after {
    content: '';
}
.img-wrapper {
  text-align: center;
}
.big-modal img {
  max-height: 60vh;
}
.img-scroller {
  overflow-x: scroll;
}
.img-scroller .img-fluid {
  max-width: initial;
}
.gridheader {
  font-size: 12px;
  margin-bottom: 10px;
  margin-left: 30px;
  margin-right: 30px;
}
.gridheader:after {
  content: '';
  display: table;
  clear: both;
}
.sortheader {
  float: right;
  cursor: default;
}
.layerinfo {
  float: left;
}
.sortby {
  text-decoration: underline;
  cursor: pointer;
}
.sortby.currentsort {
  text-decoration: none;
  font-weight: bold;
  cursor: default;
}
</style>
</head>
<body class="unitviz">
<div class="container-fluid">
'''

html_sortheader = '''
<div class="sortheader">
sort by
<span class="sortby currentsort" data-index="0">label</span>
<span class="sortby" data-index="1">score</span>
<span class="sortby" data-index="2">unit</span>
</div>
'''

html_suffix = '''
</div>
<div class="modal" id="lightbox">
  <div class="modal-dialog big-modal" role="document">
    <div class="modal-content">
      <div class="modal-header">
        <h5 class="modal-title"></h5>
        <button type="button" class="close"
             data-dismiss="modal" aria-label="Close">
          <span aria-hidden="true">&times;</span>
        </button>
      </div>
      <div class="modal-body">
        <div class="img-wrapper img-scroller">
          <img class="fullsize img-fluid">
        </div>
      </div>
      <div class="modal-footer">
        <div class="footer-caption">
        </div>
      </div>
    </div>
  </div>
</div>
<script>
$('img:not([data-nothumb])[src]').wrap(function() {
  var result = $('<a data-toggle="lightbox">');
  result.attr('href', $(this).attr('src'));
  var caption = $(this).closest('figure').find('figcaption').text();
  if (!caption && $(this).closest('.citation').length) {
    caption = $(this).closest('.citation').text();
  }
  if (caption) {
    result.attr('data-footer', caption);
  }
  var title = $(this).attr('title');
  if (!title) {
    title = $(this).closest('td').find('.unit,.score').map(function() {
      return $(this).text();
    }).toArray().join('; ');
  }
  if (title) {
    result.attr('data-title', title);
  }
  return result;
});
$(document).on('click', '[data-toggle=lightbox]', function(event) {
    var fullIndex = $(this).closest('.unit').find('.full-index').text();
    console.log('Full Index:', fullIndex); // 调试信息
    $('#lightbox img').attr('src', $(this).attr('href'));
    $('#lightbox .modal-title').text($(this).data('title') ||
       $(this).closest('.unit').find('.unitlabel').text());
    $('#lightbox .footer-caption').html(
       ( $(this).data('footer') || $(this).closest('.unit').find('.info').text() ) +
       '<br><small>Full Index: ' + fullIndex + '</small>');
    event.preventDefault();
    $('#lightbox').modal();
    $('#lightbox img').closest('div').scrollLeft(0);
});
$(document).on('keydown', function(event) {
    // Only close the modal on Escape key
    if (event.key === "Escape") {
        $('#lightbox').modal('hide');
    }
});
$(document).on('click', '.sortby', function(event) {
    var sortindex = +$(this).data('index');
    sortBy(sortindex);
    $('.sortby').removeClass('currentsort');
    $(this).addClass('currentsort');
});
function sortBy(index) {
  $('.unitgrid').find('.unit').sort(function (a, b) {
     return +$(a).eq(0).data('order').split(' ')[index] -
            +$(b).eq(0).data('order').split(' ')[index];
  }).appendTo('.unitgrid');
}
</script>
</body>
</html>
'''