"""Separate storage/API helpers for paired direction preferences, not peak labels."""
import csv
from io import StringIO
import json

CHOICES = {'', 'A', 'B', 'same', 'neither', 'unsure'}
PUBLIC_SIDE_FIELDS = ('direction', 'score', 'event_count', 'positive_votes', 'negative_votes', 'uncertain_beats', 'votes', 'decision')


def public_task(task, rating=None):
    return dict(id=task['id'], participant=task['participant'],t=task['t'],x=task['x'],
                focusStartSec=task['focusStartSec'],focusEndSec=task['focusEndSec'],
                sides={side:{key:values[key] for key in PUBLIC_SIDE_FIELDS} for side,values in task['sides'].items()},
                rating=rating)


def validate_comparison(body):
    choice = body.get('choice', '')
    status = body.get('status', '')
    confidence = body.get('confidence', 'medium')
    notes = body.get('notes', '')
    version = body.get('version')
    if not isinstance(choice,str) or choice not in CHOICES:
        raise ValueError('Choose A, B, equally reasonable, neither, or cannot judge')
    if not isinstance(status,str) or not isinstance(confidence,str) or status not in {'in_progress','complete'} or confidence not in {'high','medium','low'}:
        raise ValueError('Invalid status or confidence')
    if status == 'complete' and not choice:
        raise ValueError('Select a comparison before completing this minute')
    if not isinstance(notes,str) or len(notes)>2000:
        raise ValueError('Notes must be 2,000 characters or fewer')
    if isinstance(version,bool) or not isinstance(version,int) or version<0:
        raise ValueError('Invalid saved version')
    # Never accept a client-supplied method identity, prediction or edited waveform.
    return dict(choice=choice,status=status,confidence=confidence,notes=notes,
                review_mode='assisted_paired_direction_preference'),version


def comparison_csv(packet, ratings, csv_safe):
    buffer=StringIO(newline='')
    fields=['id','participant','choice','confidence','status','reviewer','updated_at','version','notes','review_mode']
    writer=csv.DictWriter(buffer,fieldnames=fields);writer.writeheader()
    for task in packet['tasks']:
        if task['id'] not in ratings:
            continue
        row=dict(id=task['id'],participant=task['participant'],**ratings[task['id']])
        writer.writerow({key:csv_safe(row.get(key,'')) for key in fields})
    # The local private packet resolves A/B to old/new after review. Do not
    # reveal identities to an annotator via the ordinary export endpoint.
    return buffer.getvalue().encode('utf-8-sig')
